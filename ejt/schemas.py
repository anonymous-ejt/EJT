from dataclasses import dataclass


@dataclass
class EJTPair:
    pair_id: str
    template: str
    query: str
    category: str
    subcategory: str
    generated_template: str = ""
    status: str = "pending"
