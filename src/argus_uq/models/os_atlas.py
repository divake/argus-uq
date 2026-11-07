"""OS-Atlas-Base-7B agent (kept for reference; dropped from paper headline).

OS-Atlas-Base-7B is a grounding agent from OS-Copilot. It was scored on
V2 and OSG mid-pipeline and **dropped from the paper's headline matrix**
The class is retained here so users who
want to score a fifth open-weight model on their own grounding records
can still use the agent.
"""

from __future__ import annotations

from argus_uq.models.whitebox import WhiteboxModel


class OSAtlas(WhiteboxModel):
    """OS-Atlas-Base-7B grounding agent (kept for reference; not in paper)."""

    model_id = "OS-Copilot/OS-Atlas-Base-7B"
    cell_tag = "OA"
    in_paper = False  # informational; documents that this model is not in the headline
