from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from autostats.core.schemas.stat_result import StatResult
from autostats.core.session.store import SessionStore

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ReportBuilder:
    """Renders a session's accumulated `ResultStore` (not the LLM's memory)
    into a standalone report. Plots/tables were already computed by tools at
    analysis time; this only assembles them."""

    def __init__(self, session_store: SessionStore):
        self.session_store = session_store
        self._env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

    def _results(self, session_id: str) -> list[StatResult]:
        return [r.result for r in self.session_store.get(session_id).results]

    def render_html(self, session_id: str, narrative: str | None = None) -> str:
        template = self._env.get_template("report.html.j2")
        return template.render(session_id=session_id, results=self._results(session_id), narrative=narrative)

    def build(self, session_id: str, output_path: Path, fmt: str = "html") -> Path:
        output_path = Path(output_path)
        if fmt == "html":
            output_path.write_text(self.render_html(session_id))
        elif fmt == "pdf":
            import weasyprint

            html = self.render_html(session_id)
            weasyprint.HTML(string=html).write_pdf(str(output_path))
        elif fmt == "notebook":
            self._build_notebook(session_id, output_path)
        else:
            raise ValueError(f"Unsupported report format: {fmt}")
        return output_path

    def _build_notebook(self, session_id: str, output_path: Path) -> None:
        import nbformat as nbf

        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_markdown_cell(f"# AutoStats report — session {session_id}"))
        for i, result in enumerate(self._results(session_id), start=1):
            nb.cells.append(nbf.v4.new_markdown_cell(f"## {i}. {result.test_name}\n\n{result.interpretation}"))
            for table in result.tables:
                header = "| " + " | ".join(table.columns) + " |"
                sep = "| " + " | ".join("---" for _ in table.columns) + " |"
                rows = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in table.rows)
                nb.cells.append(nbf.v4.new_markdown_cell(f"{header}\n{sep}\n{rows}"))
            for plot in result.plots:
                nb.cells.append(nbf.v4.new_markdown_cell(f"![{plot.title}]({plot.path})"))
        nbf.write(nb, str(output_path))
