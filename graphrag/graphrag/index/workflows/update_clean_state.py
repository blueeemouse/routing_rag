# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(  # noqa: RUF029
    _config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Clean the state after the update."""
    # 其实就是把增量更新过程里的中间变量删掉（主要有进行实体合并时候用到的实体ID映射，以及
    # 临时合并后的文档集合）
    logger.info("Workflow started: update_clean_state")
    keys_to_delete = [
        key_name
        for key_name in context.state
        if key_name.startswith("incremental_update_")
    ]

    for key_name in keys_to_delete:
        del context.state[key_name]

    logger.info("Workflow completed: update_clean_state")
    return WorkflowFunctionOutput(result=None)
