import numpy as np
from scipy import constants as Cn

CONSTANTS = {
    "ang_amu2SI": np.sqrt(Cn.physical_constants["atomic mass constant"][0]) * 1e-10,
    "meV2rads": 1e-3 * Cn.e / Cn.hbar,
    "MHz2meV": Cn.h * 1e6 / (Cn.e * 1e-3),
    "MHz2J": Cn.h * 1e6,
    "GHz2meV": Cn.h * 1e9 / (Cn.e * 1e-3),
    "THz2meV": Cn.h * 1e12 / (Cn.e * 1e-3),
    "meV2J": 1e-3 * Cn.e,
}

if __name__ == "__main__":
    header = f"{'Constant':<15} {'Value':<20} {'Unit'}"
    print(header)
    print("-" * len(header))

    units = {
        "ang_amu2SI": "√(kg)·m",
        "meV2rads": "rad·s⁻¹·meV⁻¹",
        "MHz2meV": "meV·MHz⁻¹",
        "MHz2J": "J·MHz⁻¹",
        "GHz2meV": "meV·GHz⁻¹",
        "THz2meV": "meV·THz⁻¹",
        "meV2J": "J·meV⁻¹",
    }

    for name, value in CONSTANTS.items():
        unit = units.get(name, "")
        print(f"{name:<15} {value:<20.6e} {unit}")
