from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy
from htmlrefine.data_pipeline.repair.strategies.bug_fix import BugFixStrategy
from htmlrefine.data_pipeline.repair.strategies.feature_complete import FeatureCompleteStrategy
from htmlrefine.data_pipeline.repair.strategies.visual_enhance import VisualEnhanceStrategy
from htmlrefine.data_pipeline.repair.strategies.holistic_rewrite import HolisticRewriteStrategy
from htmlrefine.data_pipeline.repair.strategies.polish_visual import PolishVisualStrategy
from htmlrefine.data_pipeline.repair.strategies.enhance_interaction import EnhanceInteractionStrategy
from htmlrefine.data_pipeline.repair.strategies.refine_functionality import RefineFunctionalityStrategy
from htmlrefine.data_pipeline.repair.strategies.code_cleanup import CodeCleanupStrategy
from htmlrefine.data_pipeline.repair.strategies.fix_playability import FixPlayabilityStrategy
from htmlrefine.data_pipeline.repair.strategies.fix_interaction import FixInteractionStrategy
from htmlrefine.data_pipeline.repair.strategies.visual_enrichment import VisualEnrichmentStrategy
from htmlrefine.data_pipeline.repair.strategies.fix_game import FixGameStrategy

ALL_STRATEGIES = {
    "bug_fix":              BugFixStrategy,
    "feature_complete":     FeatureCompleteStrategy,
    "visual_enhance":       VisualEnhanceStrategy,
    "holistic_rewrite":     HolisticRewriteStrategy,
    "polish_visual":        PolishVisualStrategy,
    "enhance_interaction":  EnhanceInteractionStrategy,
    "refine_functionality": RefineFunctionalityStrategy,
    "code_cleanup":         CodeCleanupStrategy,
    "fix_playability":      FixPlayabilityStrategy,
    "fix_interaction":      FixInteractionStrategy,
    "visual_enrichment":    VisualEnrichmentStrategy,
    "fix_game":             FixGameStrategy,
}

__all__ = [
    "RepairStrategy", "BugFixStrategy", "FeatureCompleteStrategy",
    "VisualEnhanceStrategy", "HolisticRewriteStrategy",
    "PolishVisualStrategy", "EnhanceInteractionStrategy",
    "RefineFunctionalityStrategy", "CodeCleanupStrategy",
    "FixPlayabilityStrategy", "FixInteractionStrategy",
    "VisualEnrichmentStrategy", "FixGameStrategy",
    "ALL_STRATEGIES",
]
