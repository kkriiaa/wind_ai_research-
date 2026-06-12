import numpy as np
import matplotlib.pyplot as plt
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane

def main():
    print("Initializing FLORIS Wind Farm...")
    
    # 1. Load the physics model and turbine configuration
    fmodel = FlorisModel("gch.yaml")
    
    # 2. Modify the layout (Place 3 turbines in a row, spaced 500 meters apart)
    fmodel.set(
        layout_x=[0.0, 500.0, 1000.0],
        layout_y=[0.0, 0.0, 0.0]
    )
    
    # 3. Set the environment conditions
    fmodel.set(
        wind_directions=[270.0],      # 270 degrees = Wind blowing West to East
        wind_speeds=[8.0],            # 8.0 meters per second
        turbulence_intensities=[0.06] # 6% ambient turbulence
    )
    
    # 4. Run the physics engine calculation
    fmodel.run()
    
    # 5. Extract the Power Output for each individual turbine
    turbine_powers = fmodel.get_turbine_powers()
    turbine_powers_mw = turbine_powers / 1_000_000 # Convert Watts to MW
    
    print("\n--- Power Output ---")
    for i, power in enumerate(turbine_powers_mw[0, :]):
        print(f"Turbine {i}: {power:.2f} MW")

    # 6. Visualize the Wake Map
    print("\nGenerating Wake Visualization...")
    # FIX: Explicitly set the height to 90.0 meters (Hub height)
    horizontal_plane = fmodel.calculate_horizontal_plane(
        height=90.0,
        x_resolution=200,
        y_resolution=100
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    visualize_cut_plane(
        horizontal_plane,
        ax=ax,
        title="FLORIS Wake Effect: 3 Turbines in a Row"
    )
    
    # Save the image directly to your folder
    plt.savefig("wake_effect.png", bbox_inches='tight')
    print("Success! Saved image to wake_effect.png")

if __name__ == "__main__":
    main()
