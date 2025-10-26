# SPDX-License-Identifier: MPL-2.0
"""
Demo script for the enhanced visualization tool.

This script demonstrates how to use the visualization tool to create various types of charts
with different configurations and output formats.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.tools.visualization import create_chart, demo_chart

def display_in_browser(image_data: str, title: str = "Chart"):
    """Display a base64 image in the default web browser."""
    # Create a simple HTML page
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f5f5f5;
            }}
            .container {{ 
                text-align: center;
                max-width: 90%;
            }}
            img {{ 
                max-width: 100%;
                height: auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            h1 {{ 
                color: #333;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <img src="{image_data}" alt="Generated chart">
        </div>
    </body>
    </html>
    """
    
    # Write to a temporary file and open in browser
    import webbrowser
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(html.encode('utf-8'))
        webbrowser.open(f"file://{f.name}")

def demo_chart(chart_type: str, data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to demonstrate a chart type."""
    print(f"\n=== {chart_type.upper()} CHART ===")
    
    # Set chart type in config
    config["chart_type"] = chart_type
    
    # Create the chart
    result = create_chart(data, config, "base64")
    
    if result.get("success", False):
        print(f"Successfully created {chart_type} chart!")
        
        # Display in browser if we have image data
        if "image_data" in result:
            display_in_browser(
                result["image_data"],
                f"{chart_type.title()} Chart: {config.get('title', '')}"
            )
    else:
        print(f"Failed to create chart: {result.get('error', 'Unknown error')}")
    
    return result

def main():
    """Run all chart demos."""
    # Sample data for all chart types
    sales_data = [
        {"x": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"], 
         "y": [10, 15, 12, 18, 22, 25, 28], 
         "label": "2023", 
         "color": "#4e79a7"}
    ]
    
    # Configuration for all charts
    base_config = {
        "title": "Sales Performance 2023",
        "x_label": "Month",
        "y_label": "Sales (in thousands)",
        "width": 12,
        "height": 6,
        "theme": "default",
        "show_legend": True,
        "grid": True,
        "legend_position": "best",
        "x_rotation": 0,
        "y_rotation": 0,
    }
    
    # Demo line chart
    print("\n=== DEMO: LINE CHART ===")
    demo_chart("line", sales_data, base_config)
    
    # Demo bar chart
    print("\n=== DEMO: BAR CHART ===")
    demo_chart("bar", sales_data, base_config)
    
    # Demo pie chart (using different data)
    pie_data = [
        {"y": 30, "label": "Electronics", "color": "#4e79a7"},
        {"y": 20, "label": "Clothing", "color": "#f28e2b"},
        {"y": 15, "label": "Furniture", "color": "#e15759"},
        {"y": 35, "label": "Other", "color": "#76b7b2"}
    ]
    pie_config = {
        "title": "Sales by Category (2023)",
        "x_label": "",
        "y_label": "",
        "width": 10,
        "height": 8,
        "theme": "default",
        "show_legend": True,
        "grid": False,
        "legend_position": "right",
        "chart_type": "pie"
    }
    print("\n=== DEMO: PIE CHART ===")
    demo_chart("pie", pie_data, pie_config)
    
    print("\nDemo complete! Chart metadata saved to 'chart_results.json'")
    print("Note: Some browsers may block popups. If charts don't appear,")
    print("check your browser's popup settings for this page.")
    print("\nTo view the charts, open the generated HTML files in your browser.")

if __name__ == "__main__":
    main()
