import streamlit as st
import plotly.express as px
import pandas as pd
import math

# --- Constants ---
GRAVITY_EARTH = 9.81
GRAVITY_MARS = 3.71
GRAVITY_MOON = 1.62

# Map planet names to their gravity values for easy lookup
PLANET_GRAVITY = {
    "Earth": GRAVITY_EARTH,
    "Mars": GRAVITY_MARS,
    "Moon": GRAVITY_MOON
}

TRAJECTORY_POINTS_COUNT = 200 # Number of points for a smooth curve

# --- Physics Engine ---
# This class is responsible for all the calculations related to projectile motion.
class PhysicsEngine:
    """
    Handles all projectile motion calculations.
    """
    def __init__(self, g: float = GRAVITY_EARTH):
        # Ensure gravity is positive upon initialization to avoid division by zero or illogical results.
        if g <= 0:
            raise ValueError("Gravity must be a positive value.")
        self.g = g # Acceleration due to gravity (m/s^2)

    def set_gravity(self, g: float):
        """Sets the gravitational acceleration, ensuring it's positive."""
        if g <= 0:
            raise ValueError("Gravity must be a positive value.")
        self.g = g

    def calculate_trajectory(self, initial_velocity: float, launch_angle_deg: float) -> list[tuple[float, float, float]]:
        """
        Calculates the trajectory points (time, x, y) for a projectile launched from (0,0).
        This function generates a series of points that define the path of the projectile.

        Args:
            initial_velocity (float): Initial speed in m/s.
            launch_angle_deg (float): Launch angle in degrees relative to the horizontal.

        Returns:
            list[tuple[float, float, float]]: A list of (time, x, y) points representing the trajectory.
        """
        # Handle edge case where gravity is zero or negative (though set_gravity should prevent this).
        # If gravity is non-positive, trajectory would be infinite or undefined, so we return a single point.
        if self.g <= 0:
            return [(0.0, 0.0, 0.0)]

        # Convert launch angle from degrees to radians for trigonometric functions
        launch_angle_rad = math.radians(launch_angle_deg)
        
        # Calculate initial velocity components in x and y directions
        v0x = initial_velocity * math.cos(launch_angle_rad)
        v0y = initial_velocity * math.sin(launch_angle_rad)

        # Calculate time of flight: the time it takes for the projectile to return to ground level (y=0).
        # This formula is derived from the vertical position equation y = v0y*t - 0.5*g*t^2.
        time_of_flight = 0.0
        if v0y >= 0: # If launched upwards or horizontally (initial vertical velocity is non-negative)
            time_of_flight = (2 * v0y) / self.g
        else: # If launched downwards (initial vertical velocity is negative)
              # For simplicity in this simulator, if launched downwards from y=0, we'll assume a short fixed duration to show movement.
              # A more complex simulation would require ground height.
              if initial_velocity > 0:
                time_of_flight = 1.0 # Arbitrary short duration for visibility if launched downwards

        # If initial velocity is zero, the projectile doesn't move, so time of flight is zero.
        if initial_velocity == 0:
            time_of_flight = 0.0

        trajectory_points = []
        # Determine the number of steps to discretize the trajectory for plotting.
        # More steps result in a smoother curve.
        num_steps = TRAJECTORY_POINTS_COUNT
        # Calculate the time step (dt) between each point. Use a small default if time_of_flight is zero.
        dt = time_of_flight / num_steps if time_of_flight > 0 else 0.1

        for i in range(num_steps + 1):
            t = i * dt
            
            # Ensure 't' does not slightly exceed the calculated time_of_flight due to floating point precision.
            if time_of_flight > 0 and t > time_of_flight:
                t = time_of_flight

            # Calculate x and y positions at time 't' using standard kinematic equations.
            x = v0x * t
            y = v0y * t - 0.5 * self.g * t**2

            # Ensure 'y' does not go below ground level (0) due to calculations or angle issues.
            if y < 0 and time_of_flight > 0:
                y = 0.0
            # Special case: if time is essentially zero and y is negative (e.g., due to initial conditions), set y to 0.
            if abs(t) < 1e-9 and y < 0:
                 y = 0.0

            trajectory_points.append((t, x, y))
            
            # Stop generating points if we've reached the exact time of flight (for positive time_of_flight).
            if time_of_flight > 0 and abs(t - time_of_flight) < 1e-9:
                break
                
        # Ensure at least one point is generated, especially for static cases (e.g., v0=0 or angle >= 90).
        if not trajectory_points:
            trajectory_points.append((0.0, 0.0, 0.0))
        return trajectory_points

    def get_flight_metrics(self, initial_velocity: float, launch_angle_deg: float) -> dict:
        """
        Calculates key flight metrics: range, maximum height, and time of flight.
        These are single scalar values derived from the initial conditions and gravity.

        Args:
            initial_velocity (float): Initial speed in m/s.
            launch_angle_deg (float): Launch angle in degrees.

        Returns:
            dict: A dictionary containing 'range', 'max_height', and 'time_of_flight'.
        """
        # Handle invalid gravity case by returning infinite metrics.
        if self.g <= 0:
            return {"range": float('inf'), "max_height": float('inf'), "time_of_flight": float('inf')}
        
        launch_angle_rad = math.radians(launch_angle_deg)
        v0x = initial_velocity * math.cos(launch_angle_rad)
        v0y = initial_velocity * math.sin(launch_angle_rad)

        time_of_flight = 0.0
        if v0y >= 0: # Upward or horizontal launch
            time_of_flight = (2 * v0y) / self.g
        else: # Downward launch or horizontal with negative velocity component
            if initial_velocity > 0:
                time_of_flight = 1.0

        # If initial velocity is zero, time of flight is zero.
        if initial_velocity == 0:
            time_of_flight = 0.0

        max_range = 0.0
        max_height = 0.0

        # Calculate range and max height only if there's a positive time of flight.
        if time_of_flight > 0:
            max_range = v0x * time_of_flight # Range = horizontal velocity * time of flight
            if v0y >= 0: # Max height is only relevant if there's upward velocity component
                max_height = (v0y**2) / (2 * self.g)
        
        # Ensure all calculated metrics are non-negative, as negative values don't make physical sense here.
        time_of_flight = max(0.0, time_of_flight)
        max_range = max(0.0, max_range)
        max_height = max(0.0, max_height)

        return {
            "range": max_range,
            "max_height": max_height,
            "time_of_flight": time_of_flight
        }

# --- Streamlit App Configuration ---
# Set the page layout to wide for better use of screen real estate and set a title for the browser tab.
st.set_page_config(layout="wide", page_title="Projectile Motion Simulator")

# Initialize session state for comparison mode
if 'trajectories' not in st.session_state:
    st.session_state.trajectories = []
if 'max_x' not in st.session_state:
    st.session_state.max_x = 100.0
if 'max_y' not in st.session_state:
    st.session_state.max_y = 50.0

# Set the main title for the Streamlit application.
st.title("Projectile Motion Simulator")

# --- Sidebar for Controls ---
# Create a sidebar for user inputs, keeping the main area clean for the plot and metrics.
st.sidebar.title("Controls")

# Comparison Mode Toggle
comparison_mode = st.sidebar.checkbox("Comparison Mode", value=False, help="Compare multiple trajectories on the same graph.")

# Slider for Initial Velocity
# Provides a visual way to adjust the initial speed of the projectile.
initial_velocity = st.sidebar.slider(
    "Initial Velocity (m/s)",
    min_value=0.0,       # Minimum value for velocity
    max_value=200.0,     # Maximum value for velocity
    value=50.0,          # Default value
    step=1.0,            # Increment step for the slider
    help="The initial speed of the projectile in meters per second."
)

# Slider for Launch Angle
# Allows users to set the launch angle from 0 to 90 degrees.
launch_angle = st.sidebar.slider(
    "Launch Angle (°)",
    min_value=0.0,       # Minimum angle
    max_value=90.0,      # Maximum angle
    value=45.0,          # Default angle
    step=1.0,            # Increment step
    help="The angle at which the projectile is launched, relative to the horizontal, in degrees."
)

# Dropdown (Selectbox) for Planet Selection
# Users can choose the celestial body to simulate under its gravitational pull.
planet = st.sidebar.selectbox(
    "Planet",
    list(PLANET_GRAVITY.keys()), # Options are the keys from the PLANET_GRAVITY dictionary
    index=0, # Default selection is the first item in the list (Earth)
    help="Select the celestial body to simulate the projectile motion under its gravity."
)

# --- Physics Calculation ---
# Get the gravity value corresponding to the selected planet.
selected_gravity = PLANET_GRAVITY[planet]
# Instantiate the PhysicsEngine with the selected gravity.
physics_engine = PhysicsEngine(g=selected_gravity)

# Calculate the flight metrics (range, max height, time of flight) based on current inputs.
metrics = physics_engine.get_flight_metrics(initial_velocity, launch_angle)
# Calculate the points that define the projectile's trajectory.
trajectory_points = physics_engine.calculate_trajectory(initial_velocity, launch_angle)

# --- Comparison Mode Controls ---
if comparison_mode:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Comparison Settings")
    
    label = st.sidebar.text_input("Trajectory Label", value=f"{planet} {initial_velocity}m/s {launch_angle}°")
    
    if st.sidebar.button("Add to Comparison"):
        df_new = pd.DataFrame(trajectory_points, columns=['Time', 'X', 'Y'])
        df_new['Label'] = label
        st.session_state.trajectories.append(df_new)
        
        # Update locked axes bounds
        current_max_x = df_new['X'].max()
        current_max_y = df_new['Y'].max()
        st.session_state.max_x = max(st.session_state.max_x, current_max_x * 1.1)
        st.session_state.max_y = max(st.session_state.max_y, current_max_y * 1.1)
        st.rerun()

    if st.sidebar.button("Clear All Trajectories"):
        st.session_state.trajectories = []
        st.session_state.max_x = 100.0
        st.session_state.max_y = 50.0
        st.rerun()

# --- Display Metric Cards ---
st.header("Flight Metrics") # Header for the metrics section
col1, col2, col3 = st.columns(3) # Create three columns for the metrics

# Display Range
col1.metric(
    "Range", # Label for the metric
    f"{metrics['range']:.2f} m", # Value, formatted to 2 decimal places
    help="The total horizontal distance covered by the projectile."
)
# Display Max Height
col2.metric(
    "Max Height",
    f"{metrics['max_height']:.2f} m",
    help="The maximum vertical altitude reached by the projectile."
)
# Display Time of Flight
col3.metric(
    "Time of Flight",
    f"{metrics['time_of_flight']:.2f} s",
    help="The total duration the projectile spends in the air."
)

# --- Plotting the Trajectory ---
st.header("Trajectory Visualization") # Header for the plot section

if trajectory_points:
    # Prepare data for plotting
    current_df = pd.DataFrame(trajectory_points, columns=['Time', 'X', 'Y'])
    current_df['Label'] = "Current"
    
    if comparison_mode and st.session_state.trajectories:
        # Combine all trajectories
        plot_df = pd.concat(st.session_state.trajectories + [current_df])
        title = "Projectile Comparison"
        color_col = 'Label'
        show_legend = True
    else:
        plot_df = current_df
        title = f'Projectile Trajectory ({planet})'
        color_col = None
        show_legend = False

    # Create plot
    fig = px.line(
        plot_df,
        x='X',
        y='Y',
        color=color_col,
        title=title,
        labels={'X': 'Horizontal Distance (m)', 'Y': 'Vertical Distance (m)'},
        hover_data={'Time': ':.2f', 'X': ':.2f', 'Y': ':.2f'}
    )
    
    # Update axes to be "locked" if comparison mode is on
    if comparison_mode:
        fig.update_xaxes(range=[0, st.session_state.max_x])
        fig.update_yaxes(range=[0, st.session_state.max_y])
    
    fig.update_layout(
        xaxis=dict(showgrid=True, zeroline=True, zerolinecolor='Gray'),
        yaxis=dict(showgrid=True, zeroline=True, zerolinecolor='Gray'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=show_legend,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Could not calculate trajectory points.")

# Informational markdown text at the bottom.
st.markdown("""
---
*Note: This simulation neglects air resistance for simplicity.*
""")
