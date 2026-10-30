"""Label pools for the synthetic training tasks. Every label returned by ranking_items is token-disjoint (split on
spaces and hyphens) from the Semantic Bandits pools (asserted). World-knowledge pools are listed per scenario index of
sbadapt_data.SCEN, from best to worst."""
import os, sys, itertools, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "CHEUNG_SBANDIT_work"))
import sbandit_harness as H
import sbadapt_data as D

WORLD_EXTRA = {
    0: [["Deep Tuna Bank", "Stone Jetty", "Empty Swimming Pool"], ["Kelp Forest Edge", "Shallow Lagoon", "Frozen Puddle"],
        ["Trout Stocked Lake", "Murky Mill Pond", "Dried Salt Flat"], ["Offshore Shipwreck", "Breakwater Wall", "Concrete Storm Drain"],
        ["Cold Upwelling Zone", "Warm Estuary", "Oil Spill Area"], ["Mangrove Channel", "Busy Ferry Dock", "Chlorinated Fountain"],
        ["Glacier Fed River", "Slow Canal", "Toxic Mine Runoff"], ["Seamount Ridge", "Sandy Beach Surf", "Parking Lot Puddle"]],
    1: [["Empty Night Expressway", "Rush-hour Ring Road", "Blocked Landslide Pass"], ["New Bypass Road", "School Zone Street", "Washed-out Bridge"],
        ["Dedicated Freight Lane", "Market Day Main Street", "Snowed-in Mountain Pass"], ["Direct Tunnel Route", "Scenic Coastal Road", "Gridlocked Bridge"],
        ["Clear Ring Motorway", "Roadworks Detour", "Riot Closed Boulevard"], ["Priority Bus Lane", "Shopping Mall Access Road", "Sinkhole Collapsed Street"],
        ["Smart Signal Corridor", "Narrow Village Lane", "Military Checkpoint Road"], ["Fresh Asphalt Highway", "Gravel Farm Road", "Mudslide Trail"]],
    2: [["Fresh Platinum Lode", "Low-grade Zinc Vein", "Flooded Empty Tunnel"], ["Thick Coal Seam", "Thin Tin Layer", "Barren Sandstone"],
        ["High-grade Silver Reef", "Weathered Bauxite Bed", "Spent Tailings Heap"], ["Untapped Lithium Brine", "Patchy Nickel Ore", "Worked-out Quarry"],
        ["Rich Uranium Pocket", "Scattered Quartz Gravel", "Dry Hole Borehole"], ["Massive Copper Porphyry", "Small Lead Lens", "Caved-in Adit"],
        ["Ruby Bearing Marble", "Grey Granite Slab", "Clay Filled Sinkhole"], ["Emerald Pegmatite", "Feldspar Outcrop", "Ash Covered Slope"]],
    3: [["Super Bowl Commercial", "Local Newspaper Ad", "Blank Billboard"], ["Viral Influencer Post", "Podcast Mid-roll", "Broken Website Banner"],
        ["Airport Terminal Screen", "Supermarket Trolley Sticker", "Faded Wall Poster"], ["Front Page Homepage Takeover", "Sidebar Text Link", "Popup Blocked Ad"],
        ["Times Square Billboard", "Subway Car Card", "Lost Leaflet"], ["Celebrity Endorsement", "Trade Show Booth", "Misprinted Coupon"],
        ["Search Engine Top Result", "Regional Magazine Page", "Unread Junk Mail"], ["Championship Halftime Spot", "Late Night Cable Ad", "Muted Radio Jingle"]],
    4: [["Liquid Cooled Supercomputer", "Mid-range Cloud Instance", "Throttled Free Tier"], ["Fresh Edge Node", "Busy Shared Cluster", "Failing Hard Drive Box"],
        ["Low Latency Metro Region", "Distant Overseas Region", "Offline Maintenance Region"], ["TPU Pod Slice", "Burstable Small Instance", "Retired Pentium Tower"],
        ["Redundant Tier IV Facility", "Single Rack Colocation", "Power Outage Site"], ["NVMe Storage Array", "Spinning Disk NAS", "Corrupted Tape Archive"],
        ["Dedicated Bare Metal", "Spot Preemptible Node", "Infected Botnet Machine"], ["Optimised Inference Cluster", "General Purpose Pool", "Overloaded Test Server"]],
    5: [["World Class Skills Clinic", "Casual Scrimmage", "Napping in Locker Room"], ["High Altitude Camp", "Treadmill Walk", "Watching Cartoons"],
        ["Video Analysis Session", "Simple Passing Drill", "Cancelled Practice"], ["Plyometric Power Circuit", "Easy Cycling", "Junk Food Party"],
        ["Pro Mentor Shadowing", "Team Yoga Class", "Phone Scrolling Break"], ["Tactical Match Simulation", "Cone Dribbling", "Bus Ride Home"],
        ["Strength Coach Program", "Warm-up Laps", "Injury Rest Day"], ["Sprint Interval Ladder", "Light Stretch Session", "Sleeping In"]],
}

PAPER_WORDS = D.PAPER_WORDS
for pools in WORLD_EXTRA.values():
    for p in pools:
        assert len(p) == 3
        for lab in p:
            for w in lab.replace("-", " ").split():
                assert w.lower() not in PAPER_WORDS, (lab, w)


# labels of sbadapt_data.py that contain a word of the paper's sentiment pools (all in the middle position)
RENAME = {"Average Choice": "Mid Choice", "Standard Grade": "Second Grade", "Moderate Payoff": "Partial Payoff",
          "Average Iron Deposit": "Modest Iron Deposit", "Standard Virtual Machine": "Shared Virtual Machine",
          "Generic Stretching Routine": "Simple Stretching Session"}
DROP_WORDS = {"average-ish"}


def tokens(label):
    return [w.lower() for w in label.replace("-", " ").split()]


def ranking_items(n_sent3=320, seed=7):
    """Every label set the v2 data generator may draw (code labels excluded): world (old + new), ordinal, sentiment triples,
    5-arm sets. Returns list of dicts {task_idx, labels (listed order), kind, k}. Every label is token-disjoint from the paper's pools."""
    rng = random.Random(seed); items = []
    ren = lambda p: [RENAME.get(x, x) for x in p]
    for si, s in enumerate(D.SCEN):
        for p in s[2] + WORLD_EXTRA[si]: items.append({"task_idx": si, "labels": ren(p), "kind": "world", "k": 3})
    for si in range(len(D.SCEN)):
        for p in D.ORD: items.append({"task_idx": si, "labels": ren(p), "kind": "ordinal", "k": 3})
        items.append({"task_idx": si, "labels": [f"Rank {i + 1} of 5" for i in range(5)], "kind": "ordinal", "k": 5})
    keep = lambda pool: [w for w in pool if w.lower() not in DROP_WORDS]
    triples = list(itertools.product(keep(D.SENT_POS), keep(D.SENT_NEU), keep(D.SENT_NEG))); rng.shuffle(triples)
    for j, t in enumerate(triples[:n_sent3]):
        items.append({"task_idx": j % len(D.SCEN), "labels": [w.capitalize() for w in t], "kind": "sentiment", "k": 3})
    bad = sorted({(l, w) for it in items for l in it["labels"] for w in tokens(l) if w in PAPER_WORDS})
    assert not bad, bad
    return items


if __name__ == "__main__":
    it = ranking_items()
    import collections
    print(len(it), collections.Counter(x["kind"] for x in it))
