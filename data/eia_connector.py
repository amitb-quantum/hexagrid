"""
Energia — EIA Grid Connector (Phase 6)
=======================================
Provides hourly electricity demand, fuel mix, and implied price data
for all major US balancing authorities via the EIA API v2.

Primary value: PJM coverage (Virginia / DC / Ohio hyperscale corridor).

Setup:
    export EIA_API_KEY=your_key_here
    python data/eia_connector.py --ba PJM
"""

import os, sys, time, warnings, argparse
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

EIA_BASE = "https://api.eia.gov/v2"

BA_INFO = {
    'PJM':  {'name': 'PJM Interconnection',   'region': 'Mid-Atlantic / Virginia'},
    'MISO': {'name': 'Midcontinent ISO',       'region': 'Midwest'},
    'CISO': {'name': 'California ISO',         'region': 'California'},
    'ERCO': {'name': 'ERCOT',                  'region': 'Texas'},
    'ISNE': {'name': 'ISO New England',        'region': 'New England'},
    'NYIS': {'name': 'New York ISO',           'region': 'New York'},
    'SWPP': {'name': 'SW Power Pool',          'region': 'Kansas / Oklahoma'},
    'BPAT': {'name': 'Bonneville Power',       'region': 'Pacific Northwest'},
    'TVA':  {'name': 'Tennessee Valley Auth',  'region': 'Southeast'},
    'FPL':  {'name': 'Florida Power & Light',  'region': 'Florida'},
}

PRICE_MODEL = {
    'floor_mwh': 25.0,
    'base_mwh':  45.0,
    'peak_mwh':  110.0,
    'low_gw':    60.0,
    'high_gw':   135.0,
}


class EIAConnector:

    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        self.api_key = api_key or os.environ.get('EIA_API_KEY', '')
        self.verbose = verbose
        if not self.api_key:
            raise ValueError(
                "EIA API key required.\n"
                "  Set env var: export EIA_API_KEY=your_key\n"
                "  Register free: https://www.eia.gov/opendata/"
            )

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [EIA] {msg}")

    def _fetch(self, path: str, extra: dict) -> list:
        """
        Fetch from EIA API v2. api_key included in params dict alongside
        bracket-keyed params — mirrors the approach from working EIA examples.
        """
        params = {'api_key': self.api_key}
        params.update(extra)
        url = f"{EIA_BASE}{path}"
        try:
            r = requests.get(url, params=params, timeout=30)
            if self.verbose:
                self._log(f"URL sent: {r.url[:130]}")
            r.raise_for_status()
            body = r.json()
            return body.get('response', {}).get('data', [])
        except Exception as e:
            self._log(f"Fetch error: {e}")
            return []

    def get_demand(self, ba: str = 'PJM', hours: int = 24) -> pd.DataFrame:
        self._log(f"Demand: {ba} ({hours}h)")
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours + 24)

        rows = self._fetch('/electricity/rto/region-data/data/', {
            'frequency':            'hourly',
            'data[0]':              'value',
            'facets[respondent][]': ba,
            'facets[type][]':       'D',
            'start':                start.strftime('%Y-%m-%dT%H'),
            'end':                  end.strftime('%Y-%m-%dT%H'),
            'sort[0][column]':      'period',
            'sort[0][direction]':   'desc',
            'length':               hours + 26,
        })

        if not rows:
            self._log(f"No rows returned for {ba}")
            return pd.DataFrame(columns=['timestamp', 'demand_gw', 'ba'])

        df = pd.DataFrame(rows)
        self._log(f"Columns: {list(df.columns)}")

        if 'value' not in df.columns:
            self._log("'value' column missing — printing first row for debug:")
            self._log(str(rows[0] if rows else "empty"))
            return pd.DataFrame(columns=['timestamp', 'demand_gw', 'ba'])

        df['timestamp'] = pd.to_datetime(df['period'], utc=True)
        df['demand_gw'] = pd.to_numeric(df['value'], errors='coerce') / 1000.0
        df['ba']        = ba
        return (
            df[['timestamp', 'demand_gw', 'ba']]
            .dropna()
            .sort_values('timestamp')
            .tail(hours)
            .reset_index(drop=True)
        )

    def get_fuel_mix(self, ba: str = 'PJM') -> pd.DataFrame:
        self._log(f"Fuel mix: {ba}")
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=26)
        records = []

        for fuel in ['NG', 'COL', 'NUC', 'WAT', 'WND', 'SUN', 'OIL', 'OTH']:
            rows = self._fetch('/electricity/rto/fuel-type-data/data/', {
                'frequency':            'hourly',
                'data[0]':              'value',
                'facets[respondent][]': ba,
                'facets[fueltype][]':   fuel,
                'start':                start.strftime('%Y-%m-%dT%H'),
                'end':                  end.strftime('%Y-%m-%dT%H'),
                'sort[0][column]':      'period',
                'sort[0][direction]':   'desc',
                'length':               2,
            })
            if rows and 'value' in rows[0]:
                records.append({'fuel': fuel, 'mw': float(rows[0].get('value') or 0)})

        return pd.DataFrame(records) if records else pd.DataFrame(columns=['fuel', 'mw'])

    def fuel_mix_summary(self, ba: str = 'PJM') -> dict:
        df = self.get_fuel_mix(ba=ba)
        if df.empty:
            return {'ba': ba, 'renewable_pct': None, 'carbon_free_pct': None,
                    'total_mw': None, 'source': 'unavailable'}
        total = df['mw'].sum()
        if total == 0:
            return {'ba': ba, 'renewable_pct': 0, 'carbon_free_pct': 0,
                    'total_mw': 0, 'source': 'EIA'}

        fmap = {'NG': 'natural_gas', 'COL': 'coal', 'NUC': 'nuclear',
                'WAT': 'hydro', 'WND': 'wind', 'SUN': 'solar',
                'OIL': 'oil', 'OTH': 'other'}
        mw = {fmap.get(r['fuel'], r['fuel']): r['mw'] for _, r in df.iterrows()}
        renewable   = mw.get('wind', 0) + mw.get('solar', 0) + mw.get('hydro', 0)
        carbon_free = renewable + mw.get('nuclear', 0)

        result = {
            'ba': ba, 'name': BA_INFO.get(ba, {}).get('name', ba),
            'region': BA_INFO.get(ba, {}).get('region', ''),
            'total_mw': round(total, 1),
            'renewable_pct':   round(renewable / total * 100, 2),
            'carbon_free_pct': round(carbon_free / total * 100, 2),
            'source': 'EIA API v2',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        for fn in ['natural_gas', 'coal', 'nuclear', 'hydro', 'wind', 'solar']:
            result[f'{fn}_mw'] = round(mw.get(fn, 0), 1)
        return result

    def implied_price_kwh(self, ba: str = 'PJM') -> dict:
        df = self.get_demand(ba=ba, hours=6)
        m  = PRICE_MODEL
        if df.empty:
            return {'current_kwh': m['base_mwh']/1000, 'current_mwh': m['base_mwh'],
                    'demand_gw': None, 'method': 'fallback_no_data'}

        gw   = float(df['demand_gw'].iloc[-1])
        pct  = max(0.0, min(1.0, (gw - m['low_gw']) / (m['high_gw'] - m['low_gw'])))
        pmwh = m['base_mwh'] + (m['peak_mwh'] - m['base_mwh']) * pct
        hour = datetime.now(timezone.utc).hour
        if 16 <= hour <= 20:   pmwh *= 1.35
        elif 7 <= hour <= 22:  pmwh *= 1.15
        pmwh = max(m['floor_mwh'], pmwh)

        return {
            'current_kwh': round(pmwh / 1000, 5),
            'current_mwh': round(pmwh, 2),
            'demand_gw':   round(gw, 2),
            'method':      'demand_correlated_estimate',
            'note':        'Estimated from EIA demand. True PJM LMP requires member account.',
        }

    def national_snapshot(self, bas=None) -> dict:
        if bas is None:
            bas = ['PJM', 'MISO', 'CISO', 'ERCO', 'ISNE', 'NYIS']
        results = {}
        for ba in bas:
            try:
                df = self.get_demand(ba=ba, hours=3)
                if not df.empty:
                    price = self.implied_price_kwh(ba=ba)
                    results[ba] = {
                        'ba': ba, 'name': BA_INFO.get(ba, {}).get('name', ba),
                        'region': BA_INFO.get(ba, {}).get('region', ''),
                        'demand_gw': round(float(df['demand_gw'].iloc[-1]), 2),
                        'price_kwh': price['current_kwh'],
                    }
                else:
                    results[ba] = {'ba': ba, 'error': 'no data'}
            except Exception as e:
                results[ba] = {'ba': ba, 'error': str(e)}
        return results

    def test(self, ba: str = 'PJM'):
        print(f"\n{'='*60}")
        print(f"  ENERGIA — EIA Connector Test  |  BA: {ba}")
        print(f"  API Key: ...{self.api_key[-6:]}")
        print(f"{'='*60}\n")

        print(f"  {'─'*50}")
        print(f"  DEMAND (last 12h)")
        t0 = time.time()
        df = self.get_demand(ba=ba, hours=12)
        print(f"  Fetch: {time.time()-t0:.1f}s")
        if not df.empty:
            print(f"  Points : {len(df)}")
            print(f"  Latest : {df['demand_gw'].iloc[-1]:.2f} GW  "
                  f"@ {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"  12h avg: {df['demand_gw'].mean():.2f} GW")
            print(f"  12h max: {df['demand_gw'].max():.2f} GW")
            print()
            for _, row in df.tail(12).iterrows():
                bar = '█' * int(row['demand_gw'] / 5)
                print(f"    {row['timestamp'].strftime('%H:%M')}  "
                      f"{row['demand_gw']:6.1f} GW  {bar}")
        else:
            print("  No data returned")

        print(f"\n  {'─'*50}")
        print(f"  IMPLIED PRICE")
        p = self.implied_price_kwh(ba=ba)
        print(f"  Demand: {p.get('demand_gw') or '?'} GW")
        print(f"  Price : ${p['current_kwh']:.5f}/kWh  ({p['current_mwh']:.2f} $/MWh)")
        print(f"  Method: {p['method']}")

        print(f"\n  {'─'*50}")
        print(f"  FUEL MIX")
        fm = self.fuel_mix_summary(ba=ba)
        if fm.get('total_mw'):
            print(f"  Total      : {fm['total_mw']:,.0f} MW")
            print(f"  Renewable  : {fm['renewable_pct']:.1f}%")
            print(f"  Carbon-free: {fm['carbon_free_pct']:.1f}%")
            for fn in ['natural_gas', 'nuclear', 'coal', 'wind', 'solar', 'hydro']:
                mw = fm.get(f'{fn}_mw', 0)
                if mw > 0:
                    print(f"  {fn:<14}: {mw:>8,.0f} MW  ({mw/fm['total_mw']*100:.1f}%)")
        else:
            print("  Fuel mix unavailable")

        print(f"\n  {'─'*50}")
        print(f"  NATIONAL SNAPSHOT")
        snap = self.national_snapshot()
        print(f"\n  {'BA':<6} {'Region':<28} {'Demand':>8}  {'Price':>10}")
        print(f"  {'─'*56}")
        for code, d in snap.items():
            if 'error' not in d:
                print(f"  {code:<6} {d['region']:<28} "
                      f"{d['demand_gw']:>6.1f} GW  ${d['price_kwh']:.4f}/kWh")
            else:
                print(f"  {code:<6} error: {d['error']}")

        print(f"\n{'='*60}")
        print(f"  Done.")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ba', default='PJM', choices=list(BA_INFO.keys()))
    args = parser.parse_args()
    try:
        EIAConnector(verbose=True).test(ba=args.ba)
    except ValueError as e:
        print(f"\n  ERROR: {e}\n")
