import contextlib
import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Iterator

import torch
from compressed_tensors.utils import disable_offloading
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import (
    dispatch_for_sequential,
    get_sequential_targets,
    handle_sequential_oom,
    trace_subgraphs,
)
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import (
    DISABLE_QAC_MODIFIERS,
    DisableQuantization,
    calibration_forward_context,
)

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["SequentialPipeline"]

_SEQ_LOG_RANK = int(os.environ.get("LOCAL_RANK", 0))


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _seq_log(msg: str) -> None:
    print(f"[{_ts()}][rank {_SEQ_LOG_RANK}][SequentialPipeline] {msg}", flush=True)


def _gpu_mem_mb() -> int:
    try:
        idx = torch.cuda.current_device()
        return torch.cuda.memory_allocated(idx) // (1024 * 1024)
    except Exception:
        return -1


def _cpu_rss_mb() -> int:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
    except Exception:
        return -1


def _get_batches(
    activations: IntermediatesCache,
    num_batches: int,
    input_names: list[str],
    desc: str,
    use_prefetch: bool = False,
) -> Iterator[tuple[int, dict]]:
    """
    Yield (batch_idx, inputs) with the next batch optionally prefetched in a
    background thread to overlap fetch (onload from offload device) with the
    main-thread forward pass.
    """
    if not use_prefetch:
        for batch_idx in tqdm(range(num_batches), desc=desc):
            inputs = activations.fetch(batch_idx, input_names)
            yield batch_idx, inputs
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for batch_idx in tqdm(range(num_batches), desc=desc):
            if future is not None:
                inputs = future.result()
            else:
                inputs = activations.fetch(batch_idx, input_names)
            if batch_idx + 1 < num_batches:
                future = executor.submit(activations.fetch, batch_idx + 1, input_names)
            else:
                future = None
            yield batch_idx, inputs


@CalibrationPipeline.register("sequential")
class SequentialPipeline(CalibrationPipeline):
    @staticmethod
    @handle_sequential_oom
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
    ):
        """
        Run a sequential data pipeline according to the following steps:

        1. The model is partitioned into subgraphs according to `sequential_targets`
        2. Data passes through each subgraph sequentially. Data is passed through each
            subgraph twice, once to trigger calibration hooks, then a second time in
            order to capture activations after quantization has occurred through hooks.
        3. The intermediate activations between each subgraph are cached and offloaded
            to the cpu between each batch in order to save memory

        This pipeline requires that the model be traceable with respect to data from the
        data loader. This may be an issue for vision models with vision datasets, due
        to specialized input processing in the model.

        In the event that tracing fails, a torch.fx.proxy.TraceError will be raised. A
        model can be made traceable by wrapping the untraceable functions (see
        llmcompressor.transformers.tracing)

        :param model: model being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        _seq_log("SequentialPipeline.__call__ START")
        t0_pipeline = datetime.datetime.now().timestamp()
        session = active_session()

        onload_device = get_main_device()
        offload_device = torch.device(dataset_args.sequential_offload_device)
        _seq_log(f"onload_device={onload_device}, offload_device={offload_device}")

        _seq_log("Calling dispatch_for_sequential ...")
        t0 = datetime.datetime.now().timestamp()
        dispatch_for_sequential(model, onload_device)
        _seq_log(f"dispatch_for_sequential done ({datetime.datetime.now().timestamp() - t0:.1f}s)")

        modifiers = session.lifecycle.recipe.modifiers
        _seq_log(f"recipe.modifiers count={len(modifiers)}, types={[type(m).__name__ for m in modifiers]}")

        _seq_log("Calling get_sequential_targets ...")
        t0 = datetime.datetime.now().timestamp()
        sequential_targets = get_sequential_targets(modifiers, model, dataset_args)
        _seq_log(
            f"get_sequential_targets done ({datetime.datetime.now().timestamp() - t0:.1f}s), "
            f"targets={sequential_targets}"
        )

        ignore = dataset_args.tracing_ignore
        _seq_log(f"tracing_ignore={ignore}")

        _seq_log("Tracing subgraphs ...")
        t0 = datetime.datetime.now().timestamp()
        sample_input = next(iter(dataloader))
        _seq_log(
            f"sample_input keys={list(sample_input.keys()) if isinstance(sample_input, dict) else type(sample_input)}"
        )
        subgraphs = trace_subgraphs(model, sample_input, sequential_targets, ignore)
        num_subgraphs = len(subgraphs)
        _seq_log(
            f"trace_subgraphs done ({datetime.datetime.now().timestamp() - t0:.1f}s), "
            f"num_subgraphs={num_subgraphs}"
        )
        for i, sg in enumerate(subgraphs):
            _seq_log(
                f"  subgraph[{i}]: input_names={sg.input_names}, "
                f"consumed_names={getattr(sg, 'consumed_names', 'N/A')}, "
                f"num_nodes={len(sg.graph.nodes) if hasattr(sg, 'graph') else 'N/A'}"
            )

        _seq_log("Calling LifecycleCallbacks.calibration_epoch_start ...")
        LifecycleCallbacks.calibration_epoch_start()

        disable_qac = any(
            type(mod).__name__ in DISABLE_QAC_MODIFIERS
            for mod in session.lifecycle.recipe.modifiers
        )
        _seq_log(
            f"disable_qac={disable_qac}, "
            f"quantization_aware_calibration={getattr(dataset_args, 'quantization_aware_calibration', False)}"
        )

        with contextlib.ExitStack() as stack:
            _seq_log("Entering calibration_forward_context ...")
            stack.enter_context(calibration_forward_context(model))
            if not dataset_args.quantization_aware_calibration or disable_qac:
                _seq_log("Entering DisableQuantization context ...")
                stack.enter_context(DisableQuantization(model))
            else:
                _seq_log("Quantization-aware calibration enabled, skipping DisableQuantization")

            _seq_log("Building IntermediatesCache from dataloader ...")
            t0 = datetime.datetime.now().timestamp()
            activations = IntermediatesCache.from_dataloader(
                dataloader, onload_device, offload_device
            )
            _seq_log(
                f"IntermediatesCache ready ({datetime.datetime.now().timestamp() - t0:.1f}s), "
                f"num_batches={len(dataloader)}"
            )

            use_loss_mask = getattr(dataset_args, "use_loss_mask", False)
            if use_loss_mask:
                _seq_log("Populating loss_masks ...")
                session.state.loss_masks = [
                    activations.fetch(batch_idx, ["loss_mask"]).get("loss_mask")
                    for batch_idx in range(len(dataloader))
                ]
                _seq_log(f"loss_masks populated, count={len(session.state.loss_masks)}")
            else:
                session.state.loss_masks = None

            for subgraph_index, subgraph in enumerate(subgraphs):
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                num_batches = len(dataloader)
                use_prefetch = getattr(dataset_args, "sequential_prefetch", False)

                _seq_log(
                    f"=== Subgraph {subgraph_index + 1}/{num_subgraphs} START "
                    f"(input_names={subgraph.input_names}, num_batches={num_batches}, "
                    f"prefetch={use_prefetch}) ==="
                )

                t0_subgraph = datetime.datetime.now().timestamp()

                with disable_offloading():
                    _seq_log(f"[Subgraph {subgraph_index + 1}] Calibration pass START")
                    t0_calib = datetime.datetime.now().timestamp()
                    for batch_idx, inputs in _get_batches(
                        activations,
                        num_batches,
                        subgraph.input_names,
                        calib_desc,
                        use_prefetch,
                    ):
                        session.state.current_batch_idx = batch_idx
                        subgraph.forward(model, **inputs)
                        if batch_idx == 0 or (batch_idx + 1) == num_batches:
                            _seq_log(
                                f"[Subgraph {subgraph_index + 1}] Calib batch {batch_idx + 1}/{num_batches} done, "
                                f"GPU={_gpu_mem_mb()}MB, RSS={_cpu_rss_mb()}MB"
                            )
                    calib_elapsed = datetime.datetime.now().timestamp() - t0_calib
                    _seq_log(f"[Subgraph {subgraph_index + 1}] Calibration pass DONE ({calib_elapsed:.1f}s)")

                    _seq_log(f"[Subgraph {subgraph_index + 1}] Calling sequential_epoch_end ...")
                    t0_epoch = datetime.datetime.now().timestamp()
                    LifecycleCallbacks.sequential_epoch_end(subgraph)
                    _seq_log(
                        f"[Subgraph {subgraph_index + 1}] sequential_epoch_end done "
                        f"({datetime.datetime.now().timestamp() - t0_epoch:.1f}s)"
                    )

                    _seq_log(f"[Subgraph {subgraph_index + 1}] Propagation pass START")
                    t0_prop = datetime.datetime.now().timestamp()
                    with HooksMixin.disable_hooks():
                        for batch_idx, inputs in _get_batches(
                            activations,
                            num_batches,
                            subgraph.input_names,
                            prop_desc,
                            use_prefetch,
                        ):
                            output = subgraph.forward(model, **inputs)
                            if subgraph_index < num_subgraphs - 1:
                                activations.update(batch_idx, output)
                                activations.delete(batch_idx, subgraph.consumed_names)
                            if batch_idx == 0 or (batch_idx + 1) == num_batches:
                                _seq_log(
                                    f"[Subgraph {subgraph_index + 1}] Prop batch {batch_idx + 1}/{num_batches} done, "
                                    f"GPU={_gpu_mem_mb()}MB, RSS={_cpu_rss_mb()}MB"
                                )
                    prop_elapsed = datetime.datetime.now().timestamp() - t0_prop
                    _seq_log(f"[Subgraph {subgraph_index + 1}] Propagation pass DONE ({prop_elapsed:.1f}s)")

                subgraph_elapsed = datetime.datetime.now().timestamp() - t0_subgraph
                _seq_log(
                    f"=== Subgraph {subgraph_index + 1}/{num_subgraphs} DONE "
                    f"(total={subgraph_elapsed:.1f}s, calib={calib_elapsed:.1f}s, prop={prop_elapsed:.1f}s) ==="
                )

        _seq_log("Calling LifecycleCallbacks.calibration_epoch_end ...")
        LifecycleCallbacks.calibration_epoch_end()

        total_elapsed = datetime.datetime.now().timestamp() - t0_pipeline
        _seq_log(f"SequentialPipeline.__call__ DONE (total={total_elapsed:.1f}s)")
