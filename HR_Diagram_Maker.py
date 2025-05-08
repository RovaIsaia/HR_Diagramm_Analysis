# Import required libraries
import pandas as pd  # For data manipulation (loading and cleaning CSV data)
import matplotlib.pyplot as plt  # For creating the H-R diagram plot
import numpy as np  # For numerical operations (e.g., log calculations)
from matplotlib.ticker import LogLocator, ScalarFormatter, FuncFormatter  # For customizing axis ticks and formatting

def load_data(file_path):
    """Load and validate the star dataset."""
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Define required columns for the H-R diagram
        required_columns = ['Temperature (K)', 'Luminosity(L/Lo)', 'Absolute magnitude(Mv)', 'Star type', 'Spectral Class']
        
        # Check if all required columns exist in the dataset
        if not all(col in df.columns for col in required_columns):
            # Raise an error if any required column is missing
            raise ValueError("Dataset must contain required columns: Temperature (K), Luminosity(L/Lo), Absolute magnitude(Mv), Star type, Spectral Class")
        
        # Return the validated DataFrame
        return df
    
    # Catch and print any exceptions (e.g., file not found, invalid CSV)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def map_star_types(df):
    """Map numeric star types to descriptive labels with accurate classifications"""
    # Define a dictionary mapping star type numbers to (label, marker size, color)
    star_type_map = {
        0: ('Red Dwarf', 30, 'red'),     # Common low-mass main sequence stars
        1: ('Brown Dwarf', 20, 'maroon'),# Failed stars that don't sustain hydrogen fusion
        2: ('White Dwarf', 25, 'green'),  # Dense stellar remnants
        3: ('Main Sequence', 50, 'blue'),# Hydrogen-burning stars (most common)
        4: ('Giant', 80, 'orange'),       # Evolved, expanded stars
        5: ('Supergiant', 100, 'purple') # Extremely massive, luminous stars
    }
    
    # Map 'Star type' column to descriptive labels (e.g., 0 → 'Red Dwarf')
    df['Star type label'] = df['Star type'].map(lambda x: star_type_map[x][0])
    
    # Map 'Star type' to marker sizes for visualization
    df['Marker size'] = df['Star type'].map(lambda x: star_type_map[x][1])
    
    # Map 'Star type' to colors for visualization
    df['Color'] = df['Star type'].map(lambda x: star_type_map[x][2])
    
    return df  # Return the modified DataFrame

def create_spectral_class_axis(ax, temp_range):
    """Create secondary x-axis (top) for Spectral Class based on temperature"""
    # Define spectral classes (hot to cool): O, B, A, F, G, K, M
    spectral_classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    
    # Define temperature boundaries for spectral classes (in Kelvin)
    temp_bounds = [40000, 30000, 10000, 7500, 6000, 3700, 2000]
    
    # Create a secondary x-axis (top) using twiny()
    ax2 = ax.twiny()
    
    # Set the secondary x-axis to logarithmic scale
    ax2.set_xscale('log')
    
    # Set the limits of the secondary x-axis to match the dataset's temperature range
    ax2.set_xlim(temp_range)
    
    # Set tick locations to spectral class boundaries
    ax2.set_xticks(temp_bounds)
    
    # Set tick labels to spectral class names
    ax2.set_xticklabels(spectral_classes)
    
    # Label the secondary x-axis
    ax2.set_xlabel('Spectral Class', fontsize=12)
    
    # Invert the secondary x-axis to match the primary (hotter on left)
    ax2.invert_xaxis()
    
    # Return the secondary axis object
    return ax2

def create_luminosity_axis(ax, mv_range):
    """Create secondary y-axis (right) for Luminosity(L/Lo) using powers of 10"""
    # Create a secondary y-axis (right) using twinx()
    ax3 = ax.twinx()
    
    # Set fixed luminosity range from 10⁻⁶ to 10⁶ (scientific notation)
    l_min = 1e-6  # Minimum luminosity (10⁻⁶)
    l_max = 1e6   # Maximum luminosity (10⁶)
    
    # Set the secondary y-axis to logarithmic scale
    ax3.set_yscale('log')
    
    # Set axis limits for luminosity
    ax3.set_ylim(l_min, l_max)

    # Define ticks at powers of 10 from 10⁻⁶ to 10⁶
    l_ticks = [10**i for i in range(-6, 7)]  # Generate 10⁻⁶, 10⁻⁵, ..., 10⁶
    l_labels = [f'$10^{{{i}}}$' for i in range(-6, 7)]  # Format as LaTeX-style exponents
    
    # Set ticks and labels for luminosity axis
    ax3.set_yticks(l_ticks)
    ax3.set_yticklabels(l_labels)

    # Add minor ticks for better readability on log scale
    ax3.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10)))
    
    # Label the secondary y-axis
    ax3.set_ylabel('Luminosity (L/Lo)', fontsize=12)
    
    return ax3  # Return the secondary axis object

def plot_hr_diagram(df):
    """Plot the H-R diagram with all required axes and visualizations"""
    # Create a new figure and axis with specified size (14x10 inches)
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Iterate over each unique star type and its corresponding color
    for s_type, color in zip(df['Star type label'].unique(), df['Color'].unique()):
        # Filter the DataFrame for the current stellar type
        subset = df[df['Star type label'] == s_type]
        
        # Check if the subset is not empty before plotting
        if not subset.empty:
            # Plot a scatter plot for the subset
            ax.scatter(subset['Temperature (K)'], 
                      subset['Absolute magnitude(Mv)'],
                      c=color,               # Use the defined color
                      s=subset['Marker size'].iloc[0],  # Use marker sizes from DataFrame
                      label=s_type,          # Set legend label
                      alpha=0.7,            # Set transparency for better visibility
                      edgecolors='black',    # Add black edges to markers
                      linewidth=0.5)         # Set edge thickness

    # Set the x-axis to logarithmic scale (temperature spans wide range)
    ax.set_xscale('log')
    
    # Invert the x-axis (hotter stars on the left)
    ax.invert_xaxis()
    
    # Invert the y-axis (lower magnitude = brighter stars)
    ax.invert_yaxis()
    
    # Label the primary x-axis
    ax.set_xlabel('Temperature (K)', fontsize=12)
    
    # Label the primary y-axis
    ax.set_ylabel('Absolute Magnitude (Mv)', fontsize=12)
    
    # Set the plot title with padding to avoid overlap
    ax.set_title('Hertzsprung-Russell Diagram', fontsize=14, pad=30)
    
    # Add a legend with font size 10 (bbox_to_anchor adjusts legend position)
    ax.legend(title='Stellar Classification', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid lines for both major and minor ticks (dashed lines with 60% opacity)
    ax.grid(True, which="both", ls="--", alpha=0.6)
    
    # Calculate temperature range with 10% buffer for visualization
    temp_min, temp_max = df['Temperature (K)'].min() * 0.9, df['Temperature (K)'].max() * 1.1
    
    # Set x-axis limits (inverted due to log scale)
    ax.set_xlim(temp_max, temp_min)
    
    # Calculate magnitude range with 1-unit buffer for visualization
    mv_min, mv_max = df['Absolute magnitude(Mv)'].min() - 1, df['Absolute magnitude(Mv)'].max() + 1
    
    # Set y-axis limits (inverted via invert_yaxis())
    ax.set_ylim(mv_max, mv_min)
    
    # Define major temperature ticks (e.g., 40000 K, 30000 K, etc.)
    major_ticks = [40000, 30000, 20000, 10000, 7500, 6000, 5000, 3000]
    
    # Set major tick locations on x-axis
    ax.set_xticks(major_ticks)
    
    # Use ScalarFormatter to display plain numbers (e.g., 2000 instead of 2e3)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    
    # Add minor ticks for finer granularity on log scale
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10))
    
    # Adjust minor tick length
    ax.tick_params(which='minor', length=4)
    
    # Add secondary spectral class axis (top x-axis)
    create_spectral_class_axis(ax, (temp_min, temp_max))
    
    # Add secondary luminosity axis (right y-axis)
    create_luminosity_axis(ax, (mv_min, mv_max))
    
    # Adjust layout to prevent overlap of labels and axes
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Reserve space for title
    
    # Display the final plot
    plt.show()

def main():
    # Specify the path to the dataset
    file_path = '6 class csv.csv'
    
    # Load the dataset
    df = load_data(file_path)
    
    # Proceed if data was loaded successfully
    if df is not None:
        # Map numeric star types to descriptive labels
        df = map_star_types(df)
        
        # Generate and display the H-R diagram
        plot_hr_diagram(df)

# Run the main function when the script is executed directly
if __name__ == '__main__':
    main()