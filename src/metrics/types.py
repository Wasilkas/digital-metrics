from functools import cached_property

from pydantic import BaseModel, ConfigDict, computed_field
from pydantic.fields import Field

from .ci import calculate_confidence_interval


class PredictMatch(BaseModel):
    """A single pred→GT matching record produced by box matching."""

    pred_label: str
    gt_label: str
    pred_index: int
    gt_index: int
    confidence: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def type(self) -> str:
        if self.pred_label == self.gt_label:
            return "TP"
        elif self.pred_label == "background":
            return "FN"
        else:
            return "FP"


class Metrics(BaseModel):
    """Per-class detection metrics."""

    model_config = ConfigDict(frozen=False)

    tp: float = Field(default=0)
    fp: float = Field(default=0)
    fn: float = Field(default=0)
    confidence: float = Field(default=0)
    ap50: float = Field(default=0)
    ap75: float = Field(default=0)
    ap50_95: float = Field(default=0)
    cohen_kappa: float = Field(default=0)

    # Private CI caches — computed once; lower/upper properties read from here.
    @cached_property
    def _precision_ci(self) -> tuple[float, float]:
        return calculate_confidence_interval(self.tp, self.tp + self.fp)

    @cached_property
    def _recall_ci(self) -> tuple[float, float]:
        return calculate_confidence_interval(self.tp, self.tp + self.fn)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1e-6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1e-6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def f1_score(self) -> float:
        return 2.0 * self.precision * self.recall / max(self.precision + self.recall, 1e-6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def perebrak(self) -> float:
        return 1.0 - self.precision

    @computed_field  # type: ignore[prop-decorator]
    @property
    def nedobrak(self) -> float:
        return 1.0 - self.recall

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision_ci_lower(self) -> float:
        return self._precision_ci[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision_ci_upper(self) -> float:
        return self._precision_ci[1]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall_ci_lower(self) -> float:
        return self._recall_ci[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall_ci_upper(self) -> float:
        return self._recall_ci[1]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def perebrak_ci_lower(self) -> float:
        return 1.0 - self.precision_ci_upper

    @computed_field  # type: ignore[prop-decorator]
    @property
    def perebrak_ci_upper(self) -> float:
        return 1.0 - self.precision_ci_lower

    @computed_field  # type: ignore[prop-decorator]
    @property
    def nedobrak_ci_lower(self) -> float:
        return 1.0 - self.recall_ci_upper

    @computed_field  # type: ignore[prop-decorator]
    @property
    def nedobrak_ci_upper(self) -> float:
        return 1.0 - self.recall_ci_lower
