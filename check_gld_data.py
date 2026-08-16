"""Standalone, read-only diagnostic script - does NOT modify any existing code or
state. Purpose: check whether Alpaca's free-tier data plan returns usable 1-minute
bars for GLD, so you can tell whether your trading window is actually covered.

Loads config/AlpacaClient exactly the way bot/main.py's run() does, then fetches
1-minute GLD bars via the same AlpacaClient.get_bars_range() method bot/backtest.py
uses, and prints a short report. Nothing here is imported by, or changes the
behavior of, bot/main.py, bot/backtest.py, or any other part of the bot.

The queried period is a fixed date range in the past (see PERIOD_START/PERIOD_END
below) rather than "the last N days relative to now", because the free Alpaca data
plan rejects recent SIP data with "subscription does not permit querying recent SIP
data" - a past period sidesteps that restriction.

Usage:
    python check_gld_data.py
"""
import sys
from datetime import datetime, timezone

from alpaca.data.timeframe import TimeFrame

from bot.alpaca_client import AlpacaClient
from config import load_config

SYMBOL = "GLD"
ASSET_CLASS = "stock"
TRADING_DAYS_WANTED = 5
PERIOD_START = datetime(2026, 7, 6, tzinfo=timezone.utc)
PERIOD_END = datetime(2026, 7, 10, 23, 59, 59, tzinfo=timezone.utc)  # end of day, so the full last trading day is included
SAMPLE_DAY = datetime(2026, 7, 9, tzinfo=timezone.utc).date()  # Thursday, a known trading day within the period
BERLIN_TZ = "Europe/Berlin"
SAMPLE_WINDOW_START = "15:29"
SAMPLE_WINDOW_END = "16:00"


def main() -> int:
    # Step 1: same config/client setup as bot/main.py's run()
    print("Lade Config und AlpacaClient (wie bot/main.py) ...")
    try:
        cfg = load_config()
        client = AlpacaClient(cfg)
    except Exception as exc:
        print(f"FEHLER beim Initialisieren von Config/AlpacaClient: {exc}")
        return 1

    # Step 2: fetch 1-minute bars for the fixed historical period via get_bars_range,
    # the same method bot/backtest.py uses for historical data.
    start = PERIOD_START
    end = PERIOD_END

    print(f"Frage 1-Minuten-Bars fuer {SYMBOL} ab: {start.date()} bis {end.date()} (UTC) ...")
    try:
        bars = client.get_bars_range(SYMBOL, TimeFrame.Minute, start, end, ASSET_CLASS)
    except Exception as exc:
        print(f"FEHLER beim Abruf der Alpaca-Daten: {exc}")
        print("Moegliche Ursachen: keine Netzwerkverbindung zu Alpaca, ungueltige/fehlende")
        print("API-Keys, oder der Datenplan deckt diesen Zeitraum/dieses Symbol nicht ab.")
        return 1

    if bars is None or bars.empty:
        print(f"Keine Daten erhalten fuer {SYMBOL} im Zeitraum {start.date()} bis {end.date()}.")
        print("Moegliche Gruende: der freie Plan liefert hier keine Minutendaten fuer GLD,")
        print("ein Feed-/Berechtigungsproblem, oder der Zeitraum faellt komplett auf")
        print("Handelsferien/Wochenenden.")
        return 1

    # Step 3: reduce to the last TRADING_DAYS_WANTED distinct trading days actually present
    unique_days = sorted({ts.date() for ts in bars.index})
    if not unique_days:
        print("Bars kamen zurueck, aber ohne auswertbare Zeitstempel - Abbruch.")
        return 1

    last_days = unique_days[-TRADING_DAYS_WANTED:]
    if len(last_days) < TRADING_DAYS_WANTED:
        print(
            f"WARNUNG: nur {len(last_days)} Handelstag(e) mit Daten gefunden "
            f"(gewuenscht: {TRADING_DAYS_WANTED}). Zeige trotzdem, was da ist."
        )

    subset = bars[bars.index.map(lambda ts: ts.date() in last_days)]

    total_bars = len(subset)
    first_ts = subset.index.min()
    last_ts = subset.index.max()
    avg_bars_per_day = total_bars / len(last_days)

    print("\n=== Ergebnis ===")
    print(f"Handelstage beruecksichtigt ({len(last_days)}): {', '.join(str(d) for d in last_days)}")
    print(f"Bars insgesamt: {total_bars}")
    print(f"Erste Bar: {first_ts}")
    print(f"Letzte Bar: {last_ts}")
    print(f"Durchschnittliche Bars pro Handelstag: {avg_bars_per_day:.1f}")

    # Step 4: sample window on a single day with confirmed data, converted to German
    # local time. The day is picked from `last_days` (real UTC trading days that
    # actually have bars) rather than from tz-converted dates: converting first and
    # then taking the latest local date can produce a phantom "day" made up only of
    # a handful of late-UTC bars that roll into the next calendar day in Berlin time
    # (e.g. a Friday's last bars showing up as "Saturday" locally, with no real data).
    if SAMPLE_DAY in last_days:
        sample_day = SAMPLE_DAY
    else:
        sample_day = last_days[-1]
        print(f"\nHINWEIS: {SAMPLE_DAY} hat keine Bars in diesem Zeitraum, verwende stattdessen {sample_day}.")

    day_bars_utc = subset[subset.index.map(lambda ts: ts.date() == sample_day)]
    try:
        day_bars = day_bars_utc.tz_convert(BERLIN_TZ)
    except Exception as exc:
        print(f"\nFEHLER bei der Zeitzonen-Umrechnung nach {BERLIN_TZ}: {exc}")
        return 1

    window_bars = day_bars.between_time(SAMPLE_WINDOW_START, SAMPLE_WINDOW_END)

    print(f"\n=== Stichprobe: {sample_day} ({BERLIN_TZ}), {SAMPLE_WINDOW_START}-{SAMPLE_WINDOW_END} Uhr ===")
    if window_bars.empty:
        print(
            f"Keine Bars zwischen {SAMPLE_WINDOW_START} und {SAMPLE_WINDOW_END} Uhr "
            f"an diesem Tag gefunden - das Handelsfenster waere mit diesen Daten NICHT abgedeckt."
        )
    else:
        print(f"{len(window_bars)} Bar(s) in diesem Fenster gefunden:")
        print(window_bars[["open", "high", "low", "close", "volume"]].to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
