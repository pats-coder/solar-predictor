# Solar Predictor (Home Assistant add-on)

Learns how **this house** produces and uses power, then forecasts solar, consumption and grid draw.
No Solcast or Forecast.Solar needed.

## How it works
- **Training data:** hourly Home Assistant statistics (Envoy production, house consumption, grid import),
  paired with Open-Meteo *archived forecasts* for the same hours, so the model learns from the same kind of
  weather input it gets live.
- **Solar model:** gradient-boosted trees on tilted irradiance for each array (north, east, west), cloud
  layers, temperature and sun position. The model learns real array size, shading and inverter behaviour
  from history, so exact panel specs aren't required.
- **Consumption model:** time of day, weekday/weekend, NSW public holidays, temperature, the past week's
  usage pattern, and forecast solar (solar-driven loads such as pool heating and EV charging).
- **EV model (optional):** EV charging is metered separately (`ev_power_stat`) and forecast with its own model using the car's charge level (`ev_soc_stat`, live `ev_soc_entity`), solar and the free window; the base-load model then learns the house without the EV.
- **Grid model:** learns hourly grid import from consumption and solar (captures sub-hourly mismatch).
  An optional hour-by-hour battery simulation applies if `battery_kwh` > 0.
- **Same-day correction:** scales the rest of today's solar by how today is tracking against forecast.
- **Accuracy:** a 28-day held-out backtest on each retrain, plus live tracking of as-issued forecasts
  (day-ahead at 9pm, same-day in the morning), compared against Solcast as a benchmark.
- Retrains daily after 2am; refreshes every 15 minutes.

## Sensors
| Entity | Meaning |
|---|---|
| `sensor.solar_predictor_solar_today` | Today's solar (actual so far + forecast). Attribute `forecast` = hourly solar, consumption and grid for today and tomorrow |
| `sensor.solar_predictor_solar_remaining_today` / `_tomorrow` / `_next_hour` | Solar forecasts (kWh) |
| `sensor.solar_predictor_consumption_today` / `_tomorrow` | Household use (kWh); attributes split base load vs EV charging |
| `sensor.solar_predictor_ev_charging_today` / `_tomorrow` | Forecast EV charging (kWh), from the charger meter, car charge level, solar and free window |
| `sensor.solar_predictor_grid_import_today` / `_tomorrow` | Grid import **outside the free window** (kWh); attribute `including_free_window` |
| `sensor.solar_predictor_next_grid_draw` | Next hour you're expected to pay for grid power |
| `sensor.solar_predictor_accuracy_*` | Backtest accuracy, % (daily totals) |
| `sensor.solar_predictor_live_accuracy_solar` / `_benchmark` | 30-day live day-ahead accuracy: this model vs Solcast |
| `sensor.solar_predictor_status` | Health, last error, training info |

Accuracy = `100 × (1 − Σ|daily error| / Σ daily actual)`.

## Install
Settings → Add-ons → Add-on Store → ⋮ → Repositories → add this repo URL → install **Solar Predictor** → Start.
