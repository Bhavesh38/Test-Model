import requests
import json
import datetime
import math
import numpy as np
from collections import OrderedDict

# ===============================================================
# COPOM IMPLIED HIKE MODEL (Python translation of CDIEPremiumNew)
# ===============================================================

class CDIEPremiumModel:
    BUSINESS_DAYS_PER_YEAR = 252

    def __init__(self):
        self.CdieRate = 14.9
        self.DollarToReal = 5.4
        self.globalError = 0.01
        self.dx = 0.00001
        self.alpha = 0.05

        # Main state dictionaries
        self.BiasContracts = {}
        self.contractExpiry = {}
        self.hikes = {}
        self.trvPositions = {}
        self.Holidays = {}
        self.liveRates = {}
        self.tickValues = {}
        self.workingDays = {}
        self.dateType = OrderedDict()
        self.CDIEPREMIUMDATA = {}

    # ----------------------------
    # Utility functions
    # ----------------------------

    def _get_bias(self, code: str) -> float:
        bias = 10
        if len(code) > 2:
            ch = code[2]
            if ch == 'F':
                bias = 100
            elif ch == 'N':
                bias = 75
            elif ch in ['J', 'V']:
                bias = 50
        # Specific exceptions
        if code in ['DIV27', 'DIJ28', 'DIV28']:
            bias = 20
        elif code == 'DIF26':
            bias = 200
        elif code in ['DIV28', 'DIF29', 'DIJ29']:
            bias = 20
        elif code in ['DIQ27', 'DIZ26', 'DIQ26', 'DIU26']:
            bias = 0
        return bias

    def _calculate_working_days(self, start: datetime.date, end: datetime.date):
        days = 0
        current = start
        while current < end:
            if current.weekday() < 5 and current.strftime("%Y-%m-%d") not in self.Holidays:
                days += 1
            current += datetime.timedelta(days=1)
        return days

    # ----------------------------
    # Main computation steps
    # ----------------------------

    def calculate_working_days_all(self):
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        today = datetime.date.today()
        self.workingDays[today_str] = 0

        # For all contracts
        for con, con_date in self.contractExpiry.items():
            end = datetime.datetime.strptime(con_date, "%Y-%m-%d").date()
            self.workingDays[con_date] = self._calculate_working_days(today, end)

        # For all meetings
        for hike_date in self.hikes.keys():
            end = datetime.datetime.strptime(hike_date, "%Y-%m-%d").date()
            self.workingDays[hike_date] = self._calculate_working_days(today, end)

    def _calculate_contract_rates(self, base_rate, hikes_dict):
        contract_rates = {}
        try:
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            amount = 1.0
            prev_wd = 0
            ind = 0
            meets = list(hikes_dict.values())

            for date_str, dtype in self.dateType.items():
                curr_wd = self.workingDays.get(date_str, 0)
                if curr_wd < 1:
                    continue

                wd_val = curr_wd - prev_wd
                if dtype == "meeting":
                    amount *= (1 + base_rate / 100.0) ** (wd_val / self.BUSINESS_DAYS_PER_YEAR)
                    base_rate += meets[ind] / 100.0
                    ind += 1
                else:
                    amount *= (1 + base_rate / 100.0) ** (wd_val / self.BUSINESS_DAYS_PER_YEAR)
                    wd_val2 = curr_wd - self.workingDays[today_str]
                    curr_val = (amount ** (self.BUSINESS_DAYS_PER_YEAR / wd_val2) - 1) * 100
                    contract_rates[date_str] = curr_val

                prev_wd = curr_wd
            return contract_rates
        except Exception as e:
            print("Error in _calculate_contract_rates:", e)
            return contract_rates

    def _calculate_normal_error(self, hikes_dict):
        err = 0.0
        contract_rates = self._calculate_contract_rates(self.CdieRate, hikes_dict)
        for _, date_str in self.contractExpiry.items():
            if self._is_after_may_2028(date_str):
                continue
            if date_str in contract_rates and date_str in self.liveRates:
                err += abs(contract_rates[date_str] - self.liveRates[date_str])
        return err

    def _is_after_may_2028(self, date_str):
        try:
            d = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            return d >= datetime.date(2029, 5, 1)
        except:
            return False

    def _get_error(self, hikes_dict, index=None, diff=0.0):
        temp = hikes_dict.copy()
        if index and index in temp:
            temp[index] += diff / 100.0

        contract_rates = self._calculate_contract_rates(self.CdieRate, temp)
        contract_count = sum(1 for d in self.contractExpiry.values() if not self._is_after_may_2028(d))
        count = 2 * contract_count + 10

        err = 0.0
        for con_code, con_date in self.contractExpiry.items():
            if self._is_after_may_2028(con_date):
                continue
            if con_date in contract_rates and con_date in self.liveRates and con_date in self.BiasContracts:
                price_diff = contract_rates[con_date] - self.liveRates[con_date]
                biasNess = count * self.BiasContracts[con_date]
                err += (price_diff ** 2) * biasNess
                if count > 1:
                    count -= 1

        diff_term = sum(abs(v) ** 2 for v in temp.values()) / 200000.0
        osc_val = self._get_oscillations(temp)
        total_error = err / max(1, contract_count) + diff_term * (osc_val + 1)
        return total_error

    def _get_oscillations(self, hikes_dict):
        dates = list(hikes_dict.keys())
        if len(dates) < 3:
            return 0
        total = 0
        prev_sign = np.sign(hikes_dict[dates[1]] - hikes_dict[dates[0]])
        min_v = hikes_dict[dates[0]]
        max_v = hikes_dict[dates[0]]
        for i in range(1, len(dates)):
            curr = hikes_dict[dates[i]]
            prev = hikes_dict[dates[i - 1]]
            curr_sign = np.sign(curr - prev)
            if curr_sign == 0:
                curr_sign = 1
            if curr_sign != prev_sign and abs(curr - (max_v if prev_sign == -1 else min_v)) > 1:
                total += 1
                min_v, max_v = curr, curr
                prev_sign = curr_sign
            if max_v < curr:
                max_v = curr
            if min_v > curr:
                min_v = curr
        return total

    def _get_derivative(self, index, hikes_dict, diff, base_err):
        return (self._get_error(hikes_dict, index, diff) - base_err) / diff

    # ----------------------------
    # Gradient Descent Loop
    # ----------------------------

    def calculate_cdie_premium(self):
        hikes_temp = self.hikes.copy()
        err = self._calculate_normal_error(hikes_temp)
        derivs = {d: 0.0 for d in hikes_temp}
        alphas = {d: self.alpha for d in hikes_temp}

        iterations = 0
        while iterations < 5000 and err >= self.globalError:
            new_rates = hikes_temp.copy()
            base_err = self._get_error(hikes_temp)

            for date in hikes_temp.keys():
                curr_deriv = self._get_derivative(date, hikes_temp, self.dx, base_err)
                if iterations == 0:
                    derivs[date] = curr_deriv
                    new_rates[date] -= curr_deriv * self.alpha
                else:
                    if curr_deriv == 0 or derivs[date] == 0:
                        continue
                    if np.sign(curr_deriv) == np.sign(derivs[date]):
                        change = abs(curr_deriv - derivs[date]) / abs(derivs[date])
                        if change < (0.1 if err <= 0.35 else 1):
                            if alphas[date] < 10000:
                                alphas[date] *= 1.2
                            elif alphas[date] < 1000000:
                                alphas[date] *= 1.1
                    else:
                        alphas[date] /= 1.75
                    new_rates[date] -= curr_deriv * alphas[date]
                    derivs[date] = curr_deriv

            new_err = self._calculate_normal_error(new_rates)
            hikes_temp = new_rates
            err = new_err
            iterations += 1

        print(f"Converged in {iterations} iterations. Error={err:.5f}")
        self.CDIEPREMIUMDATA = hikes_temp
        return hikes_temp

# ===============================================================
# Example Usage (offline demo)
# ===============================================================
if __name__ == "__main__":
    model = CDIEPremiumModel()

    # Mock data for demonstration
    model.contractExpiry = {
        "DIJ25": "2025-04-30",
        "DIZ25": "2025-12-31"
    }
    model.liveRates = {
        "2025-04-30": 10.50,
        "2025-12-31": 10.75
    }
    model.hikes = {
        "2025-03-19": 0.0,
        "2025-05-07": 0.0,
        "2025-06-18": 0.0,
        "2025-07-30": 0.0
    }
    model.dateType = OrderedDict([
        ("2025-03-19", "meeting"),
        ("2025-04-30", "contract"),
        ("2025-05-07", "meeting"),
        ("2025-06-18", "meeting"),
        ("2025-07-30", "meeting"),
        ("2025-12-31", "contract")
    ])
    model.BiasContracts = {
        "2025-04-30": 50,
        "2025-12-31": 75
    }

    # Assume no holidays for demo
    model.calculate_working_days_all()
    implied_hikes = model.calculate_cdie_premium()

    print("\n=== Implied Hikes (bps) ===")
    for date, val in implied_hikes.items():
        print(f"{date}: {val:.3f}")
