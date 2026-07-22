import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from autostats.core.schemas.stat_result import PlotArtifact


def save_current_figure(plots_dir: str, title: str) -> PlotArtifact:
    plot_id = uuid.uuid4().hex[:12]
    path = Path(plots_dir) / f"{plot_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close("all")
    return PlotArtifact(plot_id=plot_id, title=title, path=str(path))
