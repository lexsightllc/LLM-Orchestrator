"""
Data visualization tools for LLM Orchestrator.

This module provides functions to create various types of charts using matplotlib.
It's designed to be used as a standalone module without any registration requirements.
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Literal
import base64
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field, validator

class ChartType(str, Enum):
    """Supported chart types."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"

class ChartConfig(BaseModel):
    """Configuration for creating a chart."""
    title: str
    x_label: str
    y_label: str
    width: int = 10
    height: int = 6
    theme: str = "default"
    chart_type: ChartType
    show_legend: bool = True
    grid: bool = True
    legend_position: str = "best"
    x_rotation: int = 0
    y_rotation: int = 0

class VisualizationInput(BaseModel):
    """Input schema for visualization tools."""
    data: List[Dict[str, Any]]
    config: ChartConfig
    output_format: str = "base64"

    @validator('output_format')
    def validate_output_format(cls, v):
        if v not in ["base64", "file"]:
            raise ValueError("output_format must be either 'base64' or 'file'")
        return v

def create_chart(data: List[Dict[str, Any]], config: Dict[str, Any], output_format: str = "base64") -> Dict[str, Any]:
    """
    Create a chart with the given data and configuration.
    
    Args:
        data: List of data points, each with x, y, and optional styling
        config: Chart configuration
        output_format: Output format ('base64' or 'file')
    
    Returns:
        Dictionary containing the chart data and metadata
    """
    try:
        # Validate input
        vis_input = VisualizationInput(
            data=data,
            config=config,
            output_format=output_format
        )
        
        # Set the style
        plt.style.use(vis_input.config.theme)
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(vis_input.config.width, vis_input.config.height))
        
        # Plot the data based on chart type
        chart_type = vis_input.config.chart_type
        
        if chart_type == ChartType.LINE:
            for series in vis_input.data:
                ax.plot(
                    series['x'],
                    series['y'],
                    label=series.get('label'),
                    color=series.get('color'),
                    alpha=series.get('alpha', 1.0)
                )
        elif chart_type == ChartType.BAR:
            for series in vis_input.data:
                ax.bar(
                    series['x'],
                    series['y'],
                    label=series.get('label'),
                    color=series.get('color'),
                    alpha=series.get('alpha', 0.7)
                )
        elif chart_type == ChartType.PIE:
            ax.pie(
                [d['y'] for d in vis_input.data],
                labels=[d.get('label', '') for d in vis_input.data],
                colors=[d.get('color') for d in vis_input.data],
                autopct='%1.1f%%',
                startangle=90,
                shadow=True
            )
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        elif chart_type == ChartType.SCATTER:
            for series in vis_input.data:
                ax.scatter(
                    series['x'],
                    series['y'],
                    label=series.get('label'),
                    c=series.get('color'),
                    s=series.get('size', 20),
                    alpha=series.get('alpha', 0.7)
                )
        elif chart_type == ChartType.HISTOGRAM:
            for series in vis_input.data:
                ax.hist(
                    series['x'],
                    bins=series.get('bins', 10),
                    alpha=series.get('alpha', 0.7),
                    label=series.get('label'),
                    color=series.get('color')
                )
        
        # Customize the chart
        ax.set_title(vis_input.config.title)
        ax.set_xlabel(vis_input.config.x_label)
        ax.set_ylabel(vis_input.config.y_label)
        
        if vis_input.config.show_legend:
            ax.legend(loc=vis_input.config.legend_position)
        
        ax.grid(vis_input.config.grid)
        
        # Rotate x and y labels if needed
        plt.xticks(rotation=vis_input.config.x_rotation)
        plt.yticks(rotation=vis_input.config.y_rotation)
        
        # Save the plot to a temporary file or return as base64
        if vis_input.output_format == "file":
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, bbox_inches='tight', dpi=100)
                plt.close()
                return {
                    "success": True,
                    "file_path": tmp_file.name,
                    "chart_type": chart_type.value,
                    "dimensions": {"width": vis_input.config.width * 100,
                                "height": vis_input.config.height * 100}
                }
        else:
            # Save to a temporary buffer and encode as base64
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return {
                "success": True,
                "image_data": f"data:image/png;base64,{img_str}",
                "chart_type": chart_type.value,
                "dimensions": {"width": vis_input.config.width * 100,
                            "height": vis_input.config.height * 100}
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create chart: {str(e)}"
        }

def demo_chart(chart_type: str, data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to demonstrate a chart type.
    
    Args:
        chart_type: Type of chart to create (line, bar, pie, scatter, histogram)
        data: List of data points, each with x, y, and optional styling
        config: Chart configuration
        
    Returns:
        Dictionary containing the chart data and metadata
    """
    # Set chart type in config
    config["chart_type"] = chart_type
    
    # Create the chart
    result = create_chart(data, config, "base64")
    
    if result.get("success", False) and "image_data" in result:
        # Display in browser if we have image data
        import webbrowser
        from tempfile import NamedTemporaryFile
        
        # Create a temporary HTML file to display the image
        with NamedTemporaryFile(suffix='.html', delete=False) as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{config.get('title', 'Chart')}</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        margin: 40px;
                        text-align: center;
                    }}
                    h1 {{ 
                        color: #2c3e50;
                        margin-bottom: 20px;
                    }}
                    .chart-container {{
                        margin: 30px auto;
                        max-width: 900px;
                        padding: 20px;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        background: white;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        border: 1px solid #eee;
                        border-radius: 4px;
                    }}
                    .footer {{
                        margin-top: 40px;
                        color: #7f8c8d;
                        font-size: 0.9em;
                    }}
                </style>
            </head>
            <body>
                <h1>{config.get('title', 'Generated Chart')}</h1>
                <div class="chart-container">
                    <img src="{result['image_data']}" alt="Generated Chart">
                    <p>Chart type: {chart_type}</p>
                    <p>Dimensions: {result.get('dimensions', {}).get('width', 800)}x{result.get('dimensions', {}).get('height', 600)}px</p>
                </div>
                <div class="footer">
                    <p>Generated by LLM Orchestrator Visualization Tool</p>
                </div>
            </body>
            </html>
            """.encode('utf-8'))
            webbrowser.open(f"file://{f.name}")
    
    return result

def _save_plot_to_base64() -> str:
    """
    Save current matplotlib figure to base64 string.
    
    Returns:
        Base64-encoded image string
    """
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def _save_plot_to_file() -> str:
    """
    Save current matplotlib figure to a temporary file.
    
    Returns:
        Path to the saved image file
    """
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        plt.savefig(f.name, bbox_inches='tight', dpi=100)
        plt.close()
        return f.name

# Example usage:
if __name__ == "__main__":
    # Example data for a line chart
    example_data = [
        {"x": [1, 2, 3, 4, 5], 
         "y": [10, 15, 12, 18, 22], 
         "label": "Sample Data", 
         "color": "blue"}
    ]
    
    example_config = {
        "title": "Sample Chart",
        "x_label": "X Axis",
        "y_label": "Y Axis",
        "width": 10,
        "height": 6,
        "theme": "default",
        "chart_type": "line",
        "show_legend": True,
        "grid": True
    }
    
    # Create and display the chart
    result = demo_chart("line", example_data, example_config)
    print("Chart generation result:", "Success" if result.get("success") else "Failed")
