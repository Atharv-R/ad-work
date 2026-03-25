"""
Competitor ad data — synthetic generation, CSV ingestion, DB storage.

All sources normalise to CompetitorAd before hitting DuckDB,
so the NLP pipeline never cares where data came from.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from adwork.db.connection import get_db


# ── Data model (the single contract everything normalises to) ───────────

class CompetitorAd(BaseModel):
    ad_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    advertiser_name: str
    platform: str  # google, meta, amazon
    ad_copy: str
    headline: str = ""
    cta: str = ""
    category: str = ""
    first_seen: date = Field(default_factory=date.today)
    last_seen: date = Field(default_factory=date.today)
    is_active: bool = True
    spend_tier: str = "medium"  # low, medium, high


# ── Storage ─────────────────────────────────────────────────────────────

def store_competitor_ads(ads: list[CompetitorAd]) -> int:
    """Upsert competitor ads into DuckDB. Returns rows stored."""
    if not ads:
        return 0
    conn = get_db()
    rows = [
        (a.ad_id, a.advertiser_name, a.platform, a.ad_copy, a.headline,
         a.cta, a.category, str(a.first_seen), str(a.last_seen),
         a.is_active, a.spend_tier)
        for a in ads
    ]
    conn.executemany(
        """INSERT INTO competitor_ads
           (ad_id, advertiser_name, platform, ad_copy, headline,
            cta, category, first_seen, last_seen, is_active, spend_tier)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT (ad_id) DO UPDATE SET
             ad_copy=excluded.ad_copy, headline=excluded.headline,
             last_seen=excluded.last_seen, is_active=excluded.is_active,
             spend_tier=excluded.spend_tier""",
        rows,
    )
    logger.info(f"Stored {len(rows)} competitor ads")
    return len(rows)


# ── CSV / file ingestion ───────────────────────────────────────────────

_COL_ALIASES = {
    "advertiser_name": ["advertiser", "brand", "company", "competitor"],
    "platform": ["platform", "channel", "network"],
    "ad_copy": ["ad_copy", "copy", "body", "text", "description", "ad_text"],
    "headline": ["headline", "title", "ad_title"],
    "cta": ["cta", "call_to_action"],
    "category": ["category", "product_category", "vertical"],
    "spend_tier": ["spend_tier", "spend_level", "budget_tier"],
}


def ingest_competitor_csv(
    df: pd.DataFrame,
    default_platform: str = "google",
) -> list[CompetitorAd]:
    """
    Accept any reasonable CSV layout → normalise → store → return ads.
    """
    col_map: dict[str, str] = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    for field, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                col_map[field] = lower_cols[alias]
                break

    if "ad_copy" not in col_map:
        raise ValueError(
            f"CSV must have an ad copy column. "
            f"Accepted names: {_COL_ALIASES['ad_copy']}. "
            f"Got columns: {list(df.columns)}"
        )

    ads: list[CompetitorAd] = []
    for _, row in df.iterrows():
        ad = CompetitorAd(
            advertiser_name=str(row.get(col_map.get("advertiser_name", ""), "Unknown")),
            platform=str(row.get(col_map.get("platform", ""), default_platform)).lower(),
            ad_copy=str(row[col_map["ad_copy"]]),
            headline=str(row.get(col_map.get("headline", ""), "")),
            cta=str(row.get(col_map.get("cta", ""), "")),
            category=str(row.get(col_map.get("category", ""), "")),
            spend_tier=str(row.get(col_map.get("spend_tier", ""), "medium")).lower(),
        )
        if ad.ad_copy.strip():
            ads.append(ad)

    stored = store_competitor_ads(ads)
    logger.info(f"Ingested {stored} ads from CSV ({len(df)} rows input)")
    return ads


# ── Synthetic data ──────────────────────────────────────────────────────

# fmt: off
# (advertiser, category, platform, headline, ad_copy, cta, spend_tier, cluster_hint)
_SYNTHETIC_ADS: list[tuple[str, str, str, str, str, str, str, str]] = [
    # ── CLUSTER: price / deals ──────────────────────────────────────────
    ("HP", "laptops", "google", "HP Pavilion — 35% Off This Week",
     "Save big on the HP Pavilion 15 with Intel i5, 16GB RAM. Limited-time sale price $449, down from $699. Free next-day shipping on all orders.",
     "Shop Sale", "high", "price"),
    ("Logitech", "keyboards", "amazon", "Logitech MX Keys — Deal of the Day",
     "Today only: MX Keys wireless keyboard at $79.99, lowest price this year. Backlit keys, USB-C charging, pairs with 3 devices.",
     "Add to Cart", "medium", "price"),
    ("Dell", "monitors", "google", "Dell 27\" Monitor Under $200",
     "The Dell S2722QC 4K monitor is now just $189.99. IPS panel, USB-C connectivity, built-in speakers. Unbeatable value for home office setups.",
     "Buy Now", "high", "price"),
    ("Samsung", "monitors", "amazon", "Samsung Odyssey — $120 Off",
     "Curved 32\" QHD gaming monitor now $279. 165Hz refresh, 1ms response time. Prime members save an extra 5%. Deal ends Sunday.",
     "See Deal", "high", "price"),
    ("ASUS", "laptops", "google", "ASUS VivoBook — Back to School Sale",
     "ASUS VivoBook 15 OLED starting at $399. Student discount available. Ryzen 5, 8GB RAM, stunning OLED display for less.",
     "Shop Now", "medium", "price"),
    ("Logitech", "mice", "amazon", "MX Master 3S — Lightning Deal",
     "Logitech MX Master 3S ergonomic mouse at $74.99. Quiet clicks, 8K DPI sensor, MagSpeed scroll wheel. Lightning deal — 2 hours left.",
     "Claim Deal", "medium", "price"),
    ("HP", "laptops", "meta", "Flash Sale: HP Envy x360",
     "48-hour flash sale on the HP Envy x360 convertible. Was $899, now $599. Ryzen 7, 16GB, touchscreen. Don't wait.",
     "Shop Flash Sale", "high", "price"),
    ("Dell", "laptops", "google", "Dell Inspiron — Clearance Event",
     "End-of-season clearance on Dell Inspiron 14. Prices from $329 with Intel i3. Free shipping, easy returns within 30 days.",
     "View Deals", "medium", "price"),
    ("Samsung", "headphones", "meta", "Galaxy Buds2 Pro — Price Drop",
     "Samsung Galaxy Buds2 Pro now $149, down from $229. Active noise cancellation, 360 Audio, IPX7 water resistance.",
     "Get Yours", "medium", "price"),
    ("Razer", "keyboards", "amazon", "Razer Huntsman Mini — 30% Off",
     "Compact 60% gaming keyboard at its lowest price: $69.99. Optical switches, PBT keycaps, detachable USB-C cable.",
     "Buy Now", "medium", "price"),

    # ── CLUSTER: features / specs ───────────────────────────────────────
    ("Dell", "laptops", "google", "Dell XPS 15 — Power Meets Precision",
     "Intel Core i9-13900H, 32GB DDR5, 1TB SSD, 15.6\" 3.5K OLED display. CNC-machined aluminium chassis with edge-to-edge keyboard.",
     "Configure Yours", "high", "features"),
    ("ASUS", "monitors", "google", "ASUS ProArt PA279CRV — True Colour",
     "27\" 4K IPS with factory-calibrated Delta E<2. USB-C 96W delivery, 99% DCI-P3 coverage. Built for professional photo and video editing.",
     "Learn More", "high", "features"),
    ("Samsung", "laptops", "google", "Galaxy Book3 Ultra Specs",
     "13th Gen i9, RTX 4070 laptop GPU, 16\" Dynamic AMOLED 2X at 120Hz. Vapour chamber cooling, 76Wh battery, S Pen included.",
     "Explore Specs", "high", "features"),
    ("Logitech", "mice", "amazon", "MX Master 3S — Engineered for Flow",
     "8,000 DPI Darkfield sensor tracks on any surface including glass. MagSpeed electromagnetic scroll — 1,000 lines per second precision.",
     "See Features", "medium", "features"),
    ("Dell", "monitors", "amazon", "Dell UltraSharp U2723QE",
     "IPS Black technology delivers 2000:1 contrast ratio. 4K resolution, USB-C hub with 90W charging, RJ45 ethernet pass-through.",
     "View Details", "high", "features"),
    ("Razer", "keyboards", "google", "Razer BlackWidow V4 Pro",
     "Hot-swappable mechanical switches, magnetic wrist rest, command dial. Razer Chroma RGB with 16.8M colours across per-key lighting.",
     "See Full Specs", "medium", "features"),
    ("ASUS", "laptops", "amazon", "ROG Zephyrus G16 — No Compromise",
     "Intel Core Ultra 9, RTX 4090, 16\" Nebula HDR display at 240Hz. Quad speakers with Dolby Atmos, 100Wh battery, MUX switch.",
     "Check Specs", "high", "features"),
    ("Sony", "headphones", "google", "WH-1000XM5 Technical Details",
     "Two processors control 8 microphones for industry-leading noise cancellation. 30-hour battery, 3-minute quick charge gives 3 hours playback.",
     "Learn More", "high", "features"),
    ("HP", "monitors", "google", "HP Z27k G3 — Every Pixel Matters",
     "4K IPS, USB-C with 100W PD, Thunderbolt 4 daisy-chain support. Factory colour-calibrated, VESA DisplayHDR 400 certified.",
     "View Specs", "medium", "features"),
    ("Samsung", "headphones", "amazon", "Galaxy Buds3 Pro — ANC Deep Dive",
     "Dual drivers with planar tweeter deliver ultra-wide frequency response. Adaptive ANC with 3 ambient levels, 360 Audio with head tracking.",
     "Technical Details", "medium", "features"),

    # ── CLUSTER: lifestyle / brand ──────────────────────────────────────
    ("Sony", "headphones", "meta", "Hear What Matters Most",
     "The WH-1000XM5 lets you lose yourself in music and find yourself in silence. Designed for the moments between meetings, flights, and everything else.",
     "Experience Sound", "high", "lifestyle"),
    ("Bose", "headphones", "meta", "Sound Is a Feeling",
     "Bose QuietComfort Ultra headphones. Close your eyes. The world fades. Your favourite song fills the space. This is what immersive audio was meant to be.",
     "Feel the Difference", "high", "lifestyle"),
    ("Sony", "headphones", "meta", "Made for Creators Who Listen",
     "From the studio to the street, the WH-1000XM5 adapts to your world. Premium materials, all-day comfort, sound that moves with you.",
     "Discover More", "high", "lifestyle"),
    ("Bose", "speakers", "meta", "Take the Music Outside",
     "Bose SoundLink Flex — dustproof, waterproof, adventure-proof. Hangs from your backpack, fills a campsite, survives anything you throw at it.",
     "Explore", "medium", "lifestyle"),
    ("Apple", "laptops", "meta", "Create Without Boundaries",
     "MacBook Pro with M3 Max. Power that whispers. From 4K video editing to complex 3D rendering — without ever hearing the fan spin.",
     "Learn More", "high", "lifestyle"),
    ("Apple", "laptops", "meta", "Your Next Chapter Starts Here",
     "MacBook Air M3 — impossibly thin, all-day battery, Liquid Retina display. The laptop that fits your life, not the other way around.",
     "See MacBook Air", "high", "lifestyle"),
    ("Sony", "earbuds", "meta", "Your Commute, Transformed",
     "LinkBuds S weigh just 4.8g per earbud. Noise cancellation that reads your environment. Seamless switching between devices, zero friction.",
     "Transform Yours", "medium", "lifestyle"),
    ("Bose", "headphones", "meta", "Quiet Never Sounded So Good",
     "Bose 700 headphones. Eleven levels of noise cancellation. Sleek stainless steel, plush cushions, voice isolation so clear you sound like you're in a studio.",
     "Discover Quiet", "high", "lifestyle"),
    ("Dell", "laptops", "meta", "Designed for Doers",
     "The Dell XPS 13 Plus — minimalist design, maximum impact. A keyboard with zero lattice, haptic trackpad, edge-to-edge display. Simply forward.",
     "Meet XPS", "high", "lifestyle"),
    ("Samsung", "monitors", "meta", "Your View, Elevated",
     "Samsung ViewFinity S9. 5K resolution. Matte display. Built-in 4K SlimFit Camera. The creative monitor that doubles as your studio.",
     "See the View", "high", "lifestyle"),

    # ── CLUSTER: urgency / scarcity ─────────────────────────────────────
    ("Razer", "keyboards", "google", "Razer Huntsman V3 Pro — Launch Edition",
     "Only 500 Launch Edition units worldwide. Analog optical switches with adjustable actuation. Magnetic wrist rest included. Once they're gone, they're gone.",
     "Reserve Now", "high", "urgency"),
    ("Razer", "mice", "amazon", "DeathAdder V3 HyperSpeed — Selling Fast",
     "63g ultra-lightweight with 90-hour battery. Top-rated on every review site. Stock running low — only 12 left at this price.",
     "Buy Before Gone", "medium", "urgency"),
    ("Bose", "speakers", "amazon", "SoundLink Max — Pre-Order Ending",
     "Pre-order window closes Friday. Be first to experience Bose's loudest portable speaker. Stereo pairing, 20-hour battery, rope handle.",
     "Pre-Order Now", "medium", "urgency"),
    ("Dell", "laptops", "google", "XPS 15 — Inventory Alert",
     "Our most popular XPS configuration is down to final units. OLED display, i7, 32GB. Current price not guaranteed after restock.",
     "Check Availability", "high", "urgency"),
    ("ASUS", "laptops", "amazon", "ROG Strix — Limited Colour Edition",
     "Eclipse Grey colourway — exclusive to this production run. RTX 4070, 16\" 165Hz. Once this batch ships, it switches to standard black.",
     "Grab Yours", "medium", "urgency"),
    ("Samsung", "monitors", "google", "Odyssey Ark — Final Restock",
     "55\" curved 4K 165Hz. This is the last production batch before the model refresh. 3-year warranty included. Don't miss the last chance.",
     "Order Now", "high", "urgency"),
    ("Logitech", "keyboards", "amazon", "MX Mechanical — Ending Today",
     "Deal expires at midnight. Tactile Quiet switches, smart backlighting, Logi Options+ customisation. After today, back to full price.",
     "Last Chance", "medium", "urgency"),
    ("HP", "monitors", "google", "HP Omen 27qs — Low Stock Warning",
     "QHD 240Hz IPS gaming monitor. Only available through HP direct. Warehouse shows fewer than 30 units remaining at this price.",
     "Secure Yours", "medium", "urgency"),
    ("Razer", "headphones", "meta", "Razer Barracuda Pro — Drop Incoming",
     "Limited restock dropping Thursday 12pm EST. THX AAA amplified, hybrid ANC, 40-hour battery. Set your alarm — last drop sold out in 19 minutes.",
     "Set Reminder", "high", "urgency"),
    ("Apple", "laptops", "google", "MacBook Pro M3 Max — Delivery Slipping",
     "Current lead time: 3-4 weeks and growing. Order now to lock in delivery before the holiday rush. Custom configs shipping even later.",
     "Order Today", "high", "urgency"),

    # ── CLUSTER: comparison / authority ──────────────────────────────────
    ("Bose", "headphones", "google", "Bose QC Ultra — #1 Noise Cancelling",
     "Rated best-in-class noise cancellation by Wirecutter, RTINGS, and SoundGuys three years running. Better ANC than Sony, better comfort than Apple.",
     "See the Reviews", "high", "comparison"),
    ("ASUS", "laptops", "amazon", "ROG Strix vs MSI Raider — Benchmarked",
     "Independent benchmarks show 12% higher FPS in AAA titles vs the MSI Raider at the same price. Better thermals, louder speakers, longer battery.",
     "Compare Now", "medium", "comparison"),
    ("Samsung", "laptops", "google", "Galaxy Book3 vs MacBook Air — Spec Battle",
     "Bigger 15.6\" AMOLED display, more ports, expandable storage — all at $200 less than MacBook Air M2. Samsung wins on value and versatility.",
     "See Comparison", "high", "comparison"),
    ("Logitech", "mice", "amazon", "MX Master 3S — Editor's Choice 2024",
     "Winner: Tom's Hardware Editor's Choice, PCMag 4.5/5, Wirecutter Top Pick. The productivity mouse professionals choose over Apple Magic Mouse.",
     "Read Reviews", "medium", "comparison"),
    ("Dell", "monitors", "google", "UltraSharp vs LG Ergo — Head to Head",
     "Dell UltraSharp U2723QE beats the LG 27UN880 in colour accuracy (Delta E 1.2 vs 2.1), includes ethernet pass-through, and costs $50 less.",
     "Compare Models", "high", "comparison"),
    ("Bose", "speakers", "amazon", "SoundLink Flex vs JBL Flip 6",
     "Bose SoundLink Flex outperforms JBL Flip 6 in bass response and IP67 durability tests. Longer battery life at max volume (12hrs vs 10hrs).",
     "See Test Results", "medium", "comparison"),
    ("Sony", "headphones", "google", "XM5 — What the Experts Say",
     "\"The best wireless headphones you can buy\" — The Verge. \"Class-leading sound quality\" — What Hi-Fi. Over 47,000 five-star reviews on Amazon.",
     "Read Expert Reviews", "high", "comparison"),
    ("HP", "laptops", "amazon", "HP Spectre x360 — Award Winner",
     "CES Innovation Award 2024. Beats Lenovo Yoga 9i in display brightness (500 vs 400 nits), battery life (17hrs vs 14hrs), and port selection.",
     "See Awards", "medium", "comparison"),
    ("Razer", "keyboards", "google", "Huntsman V3 Pro — Tournament Proven",
     "Used by 6 of the top 10 esports teams globally. Faster actuation than any Cherry MX switch — 0.1mm analog travel vs 1.2mm mechanical.",
     "See Pro Setups", "high", "comparison"),
    ("ASUS", "monitors", "amazon", "ProArt vs BenQ SW — Colour Showdown",
     "ASUS ProArt PA32UCR delivers 97% DCI-P3 with hardware calibration at $300 less than BenQ SW321C. Same accuracy, better value.",
     "View Comparison", "high", "comparison"),
]
# fmt: on


def generate_synthetic_ads(seed: int = 42) -> list[CompetitorAd]:
    """Generate ~50 realistic competitor ads. Deterministic IDs so re-runs upsert."""
    rng = np.random.default_rng(seed)
    today = date.today()
    ads: list[CompetitorAd] = []

    for i, (adv, cat, plat, headline, copy, cta, tier, _hint) in enumerate(_SYNTHETIC_ADS):
        first = today - timedelta(days=int(rng.integers(30, 90)))
        last = today - timedelta(days=int(rng.integers(0, 7)))
        ads.append(CompetitorAd(
            ad_id=f"syn_{i:03d}",
            advertiser_name=adv,
            platform=plat,
            ad_copy=copy,
            headline=headline,
            cta=cta,
            category=cat,
            first_seen=first,
            last_seen=last,
            is_active=rng.random() > 0.15,
            spend_tier=tier,
        ))

    logger.info(f"Generated {len(ads)} synthetic competitor ads")
    return ads


def seed_competitor_data(seed: int = 42) -> int:
    """Generate synthetic ads and store them. Returns count."""
    ads = generate_synthetic_ads(seed)
    return store_competitor_ads(ads)