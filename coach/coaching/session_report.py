import os
from datetime import datetime


def generate(summary: dict, output_dir: str = ".") -> str:
    """
    Write a self-contained HTML report and return its file path.

    summary: dict returned by CoachingEngine.get_session_summary()
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.html"
    path = os.path.join(output_dir, filename)

    html = _render(summary, timestamp)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[SessionReport] Saved → {path}")
    return path


def print_summary(summary: dict):
    """Quick plain-text summary printed to stdout at session end."""
    shots = summary.get('shots', {})
    footwork = summary.get('footwork', {})
    avg_swing = summary.get('avg_swing', {})
    top_tips = summary.get('top_tips', [])

    print("\n" + "=" * 50)
    print("  SESSION SUMMARY")
    print("=" * 50)

    print(f"  Total shots : {shots.get('total', 0)}")
    for shot_type, count in shots.get('breakdown', {}).items():
        print(f"    {shot_type:<20} {count}")

    print(f"  Bounces     : {summary.get('bounces', 0)}")

    if footwork:
        print(f"\n  Footwork")
        print(f"    Idle time   : {footwork.get('idle_pct', 0):.1f}%")
        print(f"    Avg speed   : {footwork.get('avg_speed_normalized', 0):.5f} (norm)")
        print(f"    Max speed   : {footwork.get('max_speed_normalized', 0):.5f} (norm)")

    if avg_swing:
        print(f"\n  Avg swing mechanics")
        label_map = {
            'avg_elbow_angle':       'Elbow angle',
            'avg_knee_angle':        'Knee angle',
            'avg_shoulder_rotation': 'Shoulder rotation',
            'avg_arm_extension':     'Arm extension',
        }
        for key, label in label_map.items():
            if key in avg_swing:
                unit = '°' if 'angle' in key or 'rotation' in key else ''
                print(f"    {label:<22} {avg_swing[key]:.1f}{unit}")

    if top_tips:
        print(f"\n  Most common tips")
        for msg, count in top_tips:
            print(f"    [{count}×] {msg}")

    print("=" * 50 + "\n")


# ------------------------------------------------------------------ HTML render

def _render(summary: dict, timestamp: str) -> str:
    shots    = summary.get('shots', {})
    footwork = summary.get('footwork', {})
    avg_swing = summary.get('avg_swing', {})
    top_tips  = summary.get('top_tips', [])
    bounces   = summary.get('bounces', 0)

    date_str = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%B %d, %Y  %H:%M")

    # ---- shot breakdown rows ----
    breakdown_rows = ""
    for shot_type, count in shots.get('breakdown', {}).items():
        total = shots.get('total', 1) or 1
        pct = count / total * 100
        breakdown_rows += f"""
          <tr>
            <td>{shot_type}</td>
            <td>{count}</td>
            <td>
              <div class="bar-wrap">
                <div class="bar" style="width:{pct:.0f}%"></div>
              </div>
              {pct:.0f}%
            </td>
          </tr>"""

    # ---- swing metrics rows ----
    label_map = {
        'avg_elbow_angle':       ('Elbow angle',        '°',  95, 168),
        'avg_knee_angle':        ('Knee angle',          '°', 130, 165),
        'avg_shoulder_rotation': ('Shoulder rotation',   '°',  20,  90),
        'avg_arm_extension':     ('Arm extension (0-1)', '',  0.6,  1.0),
    }
    swing_rows = ""
    for key, (label, unit, lo, hi) in label_map.items():
        if key not in avg_swing:
            continue
        val = avg_swing[key]
        in_range = lo <= val <= hi
        badge = '<span class="badge good">Good</span>' if in_range else '<span class="badge warn">Check</span>'
        swing_rows += f"""
          <tr>
            <td>{label}</td>
            <td>{val:.1f}{unit}</td>
            <td>Ideal {lo}{unit} – {hi}{unit}</td>
            <td>{badge}</td>
          </tr>"""

    # ---- footwork rows ----
    idle_pct = footwork.get('idle_pct', 0)
    idle_badge = '<span class="badge good">Good</span>' if idle_pct < 50 else '<span class="badge warn">Too still</span>'
    footwork_rows = ""
    if footwork:
        footwork_rows = f"""
          <tr><td>Idle time</td><td>{idle_pct:.1f}%</td><td>{idle_badge}</td></tr>
          <tr><td>Avg speed (norm)</td><td>{footwork.get('avg_speed_normalized', 0):.5f}</td><td></td></tr>
          <tr><td>Max speed (norm)</td><td>{footwork.get('max_speed_normalized', 0):.5f}</td><td></td></tr>"""

    # ---- top tips rows ----
    tips_rows = ""
    for msg, count in top_tips:
        tips_rows += f"<tr><td>{count}×</td><td>{msg}</td></tr>"

    if not tips_rows:
        tips_rows = "<tr><td colspan='2'>No tips recorded</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tennis Session — {date_str}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f1117; color: #e0e0e0;
    padding: 32px 16px;
  }}
  .container {{ max-width: 820px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; color: #fff; }}
  .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 32px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: #1a1d27; border-radius: 12px;
    padding: 20px; border: 1px solid #2a2d3a;
  }}
  .card.full {{ grid-column: 1 / -1; }}
  h2 {{ font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: #888; margin-bottom: 14px; }}
  .stat-row {{ display: flex; justify-content: space-between;
               padding: 6px 0; border-bottom: 1px solid #2a2d3a; font-size: 0.95rem; }}
  .stat-row:last-child {{ border-bottom: none; }}
  .stat-val {{ font-weight: 600; color: #fff; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ text-align: left; color: #666; font-weight: 500;
        padding: 4px 8px 8px 0; font-size: 0.8rem; }}
  td {{ padding: 7px 8px 7px 0; border-bottom: 1px solid #2a2d3a; }}
  tr:last-child td {{ border-bottom: none; }}
  .bar-wrap {{ display: inline-block; width: 80px; height: 8px;
               background: #2a2d3a; border-radius: 4px; vertical-align: middle;
               margin-right: 6px; }}
  .bar {{ height: 100%; background: #4f8ef7; border-radius: 4px; }}
  .badge {{ font-size: 0.75rem; font-weight: 600; padding: 2px 8px;
            border-radius: 4px; }}
  .badge.good {{ background: #1a3a2a; color: #4caf7d; }}
  .badge.warn {{ background: #3a2a1a; color: #f0a050; }}
  .tip-count {{ color: #4f8ef7; font-weight: 700; white-space: nowrap; padding-right: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Tennis Session Report</h1>
  <p class="subtitle">{date_str}</p>

  <div class="grid">

    <!-- overview -->
    <div class="card">
      <h2>Overview</h2>
      <div class="stat-row"><span>Total shots</span><span class="stat-val">{shots.get('total', 0)}</span></div>
      <div class="stat-row"><span>Ball bounces detected</span><span class="stat-val">{bounces}</span></div>
    </div>

    <!-- footwork -->
    <div class="card">
      <h2>Footwork</h2>
      <table>
        <thead><tr><th>Metric</th><th>Value</th><th></th></tr></thead>
        <tbody>{footwork_rows}</tbody>
      </table>
    </div>

    <!-- shot breakdown -->
    <div class="card">
      <h2>Shot Breakdown</h2>
      <table>
        <thead><tr><th>Type</th><th>#</th><th>Share</th></tr></thead>
        <tbody>{breakdown_rows}</tbody>
      </table>
    </div>

    <!-- swing mechanics -->
    <div class="card">
      <h2>Avg Swing Mechanics</h2>
      <table>
        <thead><tr><th>Metric</th><th>Value</th><th>Range</th><th></th></tr></thead>
        <tbody>{swing_rows if swing_rows else "<tr><td colspan='4'>No shots recorded</td></tr>"}</tbody>
      </table>
    </div>

    <!-- top tips -->
    <div class="card full">
      <h2>Most Frequent Coaching Cues</h2>
      <table>
        <thead><tr><th style="width:48px">Shown</th><th>Tip</th></tr></thead>
        <tbody>
          {"".join(f'<tr><td class="tip-count">{c}×</td><td>{m}</td></tr>' for m, c in top_tips)
           if top_tips else "<tr><td colspan='2'>No tips recorded</td></tr>"}
        </tbody>
      </table>
    </div>

  </div>
</div>
</body>
</html>"""
