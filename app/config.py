"""
Central configuration: KPI definitions, dimension metadata, time period logic.
All formulas derived from full_funnel.docx business specification.
"""

from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# KPI registry
# ---------------------------------------------------------------------------

@dataclass
class KPIDef:
    name: str
    short_name: str
    numerator: str
    denominator: str
    format_pct: bool = True
    decomposition_sub_kpis: list[str] = field(default_factory=list)
    description: str = ""


KPIS: dict[str, KPIDef] = {
    "ZLUR": KPIDef(
        name="ZIP Lookup Rate",
        short_name="ZLUR",
        numerator="zip_entry",
        denominator="session",
        description="Share of sessions that viewed products (ZIP-driven intent proxy).",
    ),
    "Cart RR": KPIDef(
        name="Cart Reach Rate",
        short_name="Cart RR",
        numerator="has_cart",
        denominator="session",
        description="Share of sessions that entered the cart.",
    ),
    "SSN Submit Rate": KPIDef(
        name="SSN Submit Rate",
        short_name="SSN Submit",
        numerator="cart_ssn_done",
        denominator="has_cart",
        description="Among cart sessions, share that completed SSN/credit step.",
    ),
    "Conversion After Credit": KPIDef(
        name="Conversion After Credit",
        short_name="Post-Credit Conv",
        numerator="cart_order",
        denominator="cart_ssn_done",
        description="Among SSN-submit sessions, share that completed a cart order.",
    ),
    "Cart Conversion": KPIDef(
        name="Cart Conversion",
        short_name="Cart Conv",
        numerator="cart_order",
        denominator="has_cart",
        description="Among cart sessions, share that resulted in a cart order.",
        decomposition_sub_kpis=["SSN Submit Rate", "Conversion After Credit"],
    ),
    "Cart VC": KPIDef(
        name="Cart Visit Conversion",
        short_name="Cart VC",
        numerator="cart_order",
        denominator="session",
        description="Share of all sessions ending in a cart order.",
    ),
    "Phone RR": KPIDef(
        name="Phone Reach Rate",
        short_name="Phone RR",
        numerator="queue_call",
        denominator="session",
        description="Share of sessions that generated a queue call.",
    ),
    "Phone VC": KPIDef(
        name="Phone Visit Conversion",
        short_name="Phone VC",
        numerator="phone_order",
        denominator="session",
        description="Share of sessions resulting in a phone order.",
    ),
    "VC": KPIDef(
        name="Visit Conversion (Total)",
        short_name="VC",
        numerator="total_order",        # cart_order + phone_order
        denominator="session",
        description="Share of all sessions resulting in any order (cart + phone).",
    ),
}

# ---------------------------------------------------------------------------
# Dimensions available for slicing / driver analysis
# ---------------------------------------------------------------------------

DRIVER_DIMENSIONS = [
    "website",
    "marketing_channel",
    "mover_switcher",
    "device_type",
    "landing_page_type",
    "first_partner_name",
]

DIMENSION_DISPLAY_NAMES = {
    "website": "Site",
    "marketing_channel": "Channel",
    "mover_switcher": "Mover/Switcher",
    "device_type": "Device",
    "landing_page_type": "Landing Page",
    "first_partner_name": "Entry Partner",
    "_initiative_label": "Initiative",
}

INITIATIVE_COLUMNS = {
    "is_fmp": "FMP",
    "is_lp": "LP JO Only",
    "is_grid": "Grid JO Only",
    "grid_lp": "LP + Grid JO",
    "isHoldout": "Model vs Holdout",
}

DEFAULT_CHANNELS = ["Paid Search", "Direct", "Organic", "pMax"]

CREDIT_QUALITY_FLAGS = [
    "cart_credit_fail",
    "cart_provider_pass",
    "cart_volt_fail",
    "cart_qual_fail",
]

# ---------------------------------------------------------------------------
# Time period helpers
# ---------------------------------------------------------------------------

TIME_MODES = {
    "WoW": "Week over Week (full Mon-Sun vs prior Mon-Sun)",
    "WTD (Tue review)": "Mon vs prior Mon (Tuesday morning review)",
    "WTD (Fri review)": "Mon-Thu vs prior Mon-Thu (Friday review)",
    "MoM": "Month over Month (calendar month vs prior calendar month)",
    "Custom": "Custom date ranges",
}
