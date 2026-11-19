"""Publication metadata layered on the frozen source registry."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
PROVENANCE_PATH = ROOT / "data/annotations/manuscript_provenance_user_update_20261119.json"
PROVENANCE = json.loads(PROVENANCE_PATH.read_text())
AUTHORSHIP = json.loads((BASE / "AUTHORSHIP.json").read_text())
AUTHORS = AUTHORSHIP["authors"]
AFFILIATIONS = AUTHORSHIP["affiliations"]
CORRESPONDING = next(author for author in AUTHORS if author["corresponding"])

ORIGINS_EN = {
    "ZY01": "Yan’an, Shaanxi", "ZY02": "Yunnan", "ZY03": "Inner Mongolia",
    "ZY04": "Lingshi, Shanxi", "ZY05": "Bozhou, Anhui", "ZY06": "Shijiazhuang, Hebei",
    "HQ01": "Qujing, Yunnan", "HQ02": "Guyang, Inner Mongolia", "HQ03": "Gansu",
    "HQ04": "Datong, Shanxi", "ZR01": "Yulin, Shaanxi", "ZR02": "Liaoning",
    "ZR03": "Inner Mongolia", "TM01": "Leishan, Guizhou", "TM02": "Changbai Mountain, Jilin",
    "TM04": "Yunnan",
}
for group, alias in PROVENANCE["origin_display_updates"].items():
    ORIGINS_EN[group] = alias["publication_en"]

AUTHOR_NAMES = [author["name"] for author in AUTHORS]
AUTHOR_APPROVAL = "All authors have read and approved the manuscript and its submission to Applied Spectroscopy."
AUTHOR_APPROVAL_ZH = "所有作者均已审阅并批准稿件及其向 Applied Spectroscopy 投稿。"
AI_DISCLOSURE = "OpenAI Codex assisted with revising code, compiling research materials, checking journal submission policies, and refining the manuscript."
AI_DISCLOSURE_ZH = "OpenAI Codex 辅助修改代码、汇总资料、查询期刊投稿政策和优化论文内容。"
CONFLICTS = "The authors declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article."
CONFLICTS_ZH = "作者声明，本研究、论文署名及发表均无潜在利益冲突。"
PENDING_CONFLICTS = [author["name"] for author in AUTHORS if author["competing_interest"] is None]
if PENDING_CONFLICTS:
    CONFLICTS = "The corresponding author will provide the complete competing-interest declarations at submission."
    CONFLICTS_ZH = "通讯作者将在投稿时提供完整的利益冲突声明。"
DATA_PROVENANCE = (
    "Yiyi Yang acquired and shared the spectral records. Yin Shen provided the dataset for this study "
    "as part of a shared research resource also used in other supervised studies."
)
DATA_PROVENANCE_ZH = "杨易易采集并公开分享光谱数据。Yin Shen 提供了本研究所用的数据，这些数据也用于其他指导研究。"
