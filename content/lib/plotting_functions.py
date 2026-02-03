


import matplotlib.pyplot as plt

def plot_pressure_timecourses(times_s, glucose_mg_ml, pressure, rep=0, title=None):
    P = pressure[rep]
    plt.figure()
    for j, g in enumerate(glucose_mg_ml):
        plt.plot(times_s, P[:, j], marker="o", label=f"{g:g} mg/mL")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (mm H2O)")
    plt.title(title or "Synthetic CO₂ pressure time-courses")
    plt.legend()
    plt.show()
    
    
    
    