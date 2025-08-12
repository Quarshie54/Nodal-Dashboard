import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import psapy.BeggsandBrill as BB


print("psapy imported successfully!")

import numpy as np
import psapy.BeggsandBrill as BB 
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Tuple, List, Dict, Optional
import os

# setting up Streamlit page
st.set_page_config(page_title="NODAL Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("Production Well Evaluation Dashboard")

# Constants & Defaults

MAX_PRESSURE = 10000  # psi 
MIN_TUBING_DIAM = 1.0  # inches
MAX_TUBING_DIAM = 6.0  # inches

DEFAULT_PARAMS = {'oil_rate': 6500.0,'water_rate': 150.0,'gor': 1200.0,'gas_grav': 0.65,'oil_grav': 35.0,'wtr_grav': 1.07,'diameter': 2.441,'angle': 90.0,
                  'thp': 350.0,'tht': 100.0,'twf': 150.0,'depth': 12000.0,'pres': 6000.0,'pi': 5.0}
    
# Session State Initialization
st.session_state.setdefault('pressure_data', {})
st.session_state.setdefault('nodal_data', {})

# Sidebar Configuration

with st.sidebar:
    st.header("⚙️ Well Parameters")
    
    # Fluid Properties
    with st.expander("Fluid Properties", expanded=True):
        oil_grav = st.number_input("API Gravity", min_value=9.0, max_value=50.0, 
                                  value=DEFAULT_PARAMS['oil_grav'], help="Oil API gravity (10-50)")
        gas_grav = st.number_input("Gas Gravity (air=1)", 0.5, 1.5, DEFAULT_PARAMS['gas_grav'], 
                           0.01, help="Gas specific gravity relative to air")
        wtr_grav = st.number_input("Water Gravity", 1.0, 1.2, DEFAULT_PARAMS['wtr_grav'], 
                                 0.01, help="Water specific gravity (typically 1.0-1.1)")
        water_cut = st.number_input("Water Cut (%)", 0.0, 100.0, 2.3, 0.1, 
                            help="Percentage of water in total liquid")
    
    # Well Configuration
    with st.expander("Well Configuration", expanded=True):
        diameter = st.number_input("Tubing ID (inches)", MIN_TUBING_DIAM, MAX_TUBING_DIAM, 
                           DEFAULT_PARAMS['diameter'], 0.1, 
                           help="Internal diameter of production tubing")
        angle = st.number_input("Well Deviation (deg)", -90.0, 90.0, DEFAULT_PARAMS['angle'], 
                        1.0, help="Well inclination from vertical (0=vertical)")
        depth = st.number_input("Total Depth (ft)", 1000, 30000, int(DEFAULT_PARAMS['depth']),
                              help="Measured depth from surface to bottomhole")
    
    # Reservoir Parameters
    with st.expander("Reservoir Parameters", expanded=True):
        pres = st.number_input("Reservoir Pressure (psi)", 100, 20000, int(DEFAULT_PARAMS['pres']),
                             help="Current reservoir pressure at datum depth")
        pi = st.number_input("Productivity Index (PI)", 0.1, 20.0, DEFAULT_PARAMS['pi'], 0.1,
                          help="Barrels per day per psi drawdown (STB/d/psi)")
        oil_rate = st.number_input("Oil Rate (STB/d)", 100, 20000, int(DEFAULT_PARAMS['oil_rate']),
                                 help="Current oil production rate")
    
    # Operating Conditions
    with st.expander("Operating Conditions", expanded=True):
        thp = st.number_input("Tubing Head Pressure (psi)", 0, 5000, int(DEFAULT_PARAMS['thp']),
                            help="Pressure at wellhead")
        tht = st.number_input("Tubing Head Temp (°F)", 50.0, 300.0, DEFAULT_PARAMS['tht'], 1.0,
                           help="Temperature at wellhead")
        twf = st.number_input("Bottomhole Temp (°F)", 100.0, 400.0, DEFAULT_PARAMS['twf'], 1.0,
                            help="Temperature at bottomhole")
        gor = st.number_input("Gas Oil Ratio (SCF/STB)", 0, 5000, int(DEFAULT_PARAMS['gor']),
                           help="Gas-oil ratio at current conditions")
    
    # Additional Controls for resetting and clearing cache
    st.divider()
    st.caption("System Controls")
    if st.button("Reset to Defaults"):
        st.session_state.clear()
        st.rerun()
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.session_state.clear()
        st.session_state.setdefault('pressure_data', {})
        st.session_state.setdefault('nodal_data', {})
        st.success("Cache cleared successfully!")

# Calculating water rate from water cut
water_rate = (oil_rate * water_cut) / 100.0 if water_cut > 0.0 else 0.0


# Core Calculation Functions (Cached)

@st.cache_data(show_spinner="Calculating temperature profile...")
def temp_gradient(t0: float, t1: float, depth: float) -> float:
    """Calculate geothermal gradient with validation"""
    if depth <= 0:
        return 0.0
    return abs(t1 - t0) / depth

@st.cache_data(show_spinner="Computing pressure traverse...")
def calculate_pressure_traverse(params: Dict[str, float]) -> Tuple[List[float], List[str], np.ndarray, np.ndarray]:
    """Calculate pressure profile with enhanced physics validation"""
    # Validate input parameters
    if params['depth'] <= 0:
        st.error("Well depth must be positive")
        return [params['thp']], ['Surface'], np.array([0]), np.array([params['tht']])
    
    # Calculate temperature profile
    t_grad = temp_gradient(params['tht'], params['twf'], params['depth'])
    depths = np.linspace(0, params['depth'], 51)
    temps = params['tht'] + t_grad * depths
    
    # Initialize results
    p = [max(0, min(params['thp'], MAX_PRESSURE))]  # Clamp to physical limits
    patterns = ['Surface']
    
    # Iterate through depth points
    for i in range(1, len(depths)):
        try:
            dz = depths[i] - depths[i-1]
            
            # Validate input to BB.Pgrad
            current_pressure = p[i-1]
            if current_pressure <= 0 or current_pressure > MAX_PRESSURE:
                raise ValueError(f"Invalid pressure {current_pressure} psi at depth {depths[i-1]} ft")
            
            # Call BB.Pgrad with validation
            result = BB.Pgrad(
                current_pressure, temps[i], params['oil_rate'], params['water_rate'], params['gor'],
                max(0.3, min(params['gas_grav'], 2.0)),  # Clamp to reasonable range
                max(10.0, min(params['oil_grav'], 50.0)),
                max(1.0, min(params['wtr_grav'], 1.2)),
                max(MIN_TUBING_DIAM, min(params['diameter'], MAX_TUBING_DIAM)),
                max(-90.0, min(params['angle'], 90.0)))
            
            # Handling BB.Pgrad results
            if isinstance(result, tuple) and len(result) >= 3:
                dpdz, holdup, pattern_code = result
                pattern_map = {1: 'Segregated',2: 'Transition',3: 'Intermittent',4: 'Distributed'}
                pattern_name = pattern_map.get(int(pattern_code), 'Unknown')
            else:
                dpdz = float(result)
                pattern_name = 'Unknown'
            
            # Calculate new pressure with constraints
            new_pressure = current_pressure + dpdz * dz
            new_pressure = max(0, min(new_pressure, MAX_PRESSURE))  # Apply physical limits
            p.append(new_pressure)
            patterns.append(pattern_name)
            
        except Exception as e:
            # Fallback to previous value with error marker
            p.append(p[i-1])
            patterns.append(f'Error: {str(e)[:30]}')
    
    return p, patterns, depths, temps

@st.cache_data(show_spinner="Generating VLP curve...")
def calculate_vlp(rates: List[float], params: Dict[str, float]) -> List[float]:
    """Calculate Vertical Lift Performance with progress"""
    bhps = []
    progress_bar = st.progress(0)
    total = len(rates)
    
    for idx, q in enumerate(rates):
        # Update parameters for this rate
        rate_params = params.copy()
        rate_params['oil_rate'] = q
        
        # Calculate pressure profile
        p_profile, _, _, _ = calculate_pressure_traverse(rate_params)
        bhps.append(p_profile[-1])
        
        # Update progress
        progress = int((idx + 1) / total * 100)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    return bhps

def calculate_ipr(rates: List[float], pi: float, pres: float) -> List[float]:
    """Calculate Inflow Performance Relationship with validation"""
    if pi <= 0:
        return [pres] * len(rates)  # Fallback for invalid PI
    return [max(0, pres - (rate / pi)) for rate in rates]

def find_intersection(x: List[float], y1: List[float], y2: List[float]) -> Optional[float]:
    try:
        # Convert to numpy arrays
        arr_y1 = np.array(y1)
        arr_y2 = np.array(y2)
        
        # Find where the difference changes sign
        diff = arr_y1 - arr_y2
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        
        if len(sign_changes) > 0:
            # Find the first valid intersection
            for idx in sign_changes:
                if idx < len(x) - 1:
                    # Linear interpolation for accuracy
                    x0, x1 = x[idx], x[idx+1]
                    y0, y1 = diff[idx], diff[idx+1]
                    
                    if y0 == y1:
                        return x0
                    
                    # Linear interpolation: y = m(x - x0) + y0
                    x_intersect = x0 - y0 * (x1 - x0) / (y1 - y0)
                    return float(x_intersect)
        return None
    except Exception:
        return None



# Main Calculations
# Prepare parameters dictionary with validation
params = {'oil_rate': float(oil_rate),'water_rate': float(water_rate),'gor': float(gor),'gas_grav': float(gas_grav),'oil_grav': float(oil_grav),
          'wtr_grav': float(wtr_grav),'diameter': float(diameter),'angle': float(angle),'thp': float(thp),'tht': float(tht),'twf': float(twf),'depth': float(depth)}

# Calculate pressure traverse with error handling
try:
    with st.spinner("Calculating wellbore profiles..."):
        p, patterns, depths, temps = calculate_pressure_traverse(params)
except Exception as e:
    st.error(f"Pressure calculation failed: {str(e)}")
    st.stop()

# Calculate VLP and IPR
try:
    with st.spinner("Generating performance curves..."):
        rate_range = list(range(100, 13000, 500))
        bhps = calculate_vlp(rate_range, params)
        pwfs = calculate_ipr(rate_range, float(pi), float(pres))
        optimal_rate = find_intersection(rate_range, bhps, pwfs)
except Exception as e:
    st.error(f"Performance curve calculation failed: {str(e)}")
    st.stop()

# Create pattern table
pattern_df = pd.DataFrame({"Depth (ft)": depths,"Pressure (psi)": p,"Temperature (°F)": temps,"Flow Pattern": patterns})

# Dashboard Visualization
# Main tabs
tab1, tab2, tab3, = st.tabs(['📊 Wellbore Profiles', '⚡ Nodal Analysis', '🌊 Flow Patterns'])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pressure Profile")
        fig_pressure = go.Figure()
        fig_pressure.add_trace(go.Scatter(x=p, y=depths,mode='lines+markers', name='Pressure',line=dict(color='red', width=2),hovertemplate='<b>%{x:.0f} psi</b> at %{y:.0f} ft'))

        fig_pressure.update_layout(yaxis_title='Depth (ft)',xaxis_title='Pressure (psi)',yaxis=dict(autorange='reversed', title_font=dict(family= 'Times New Roman', color='black', size=20), tickfont=dict(color='black', size=20)), 
                                   xaxis=dict(title_font=dict(family= 'Times New Roman', color='black', size=20), tickfont=dict(color='black', size=20)), height=600,hovermode='x unified')                   
        st.plotly_chart(fig_pressure, use_container_width=True)

        # Pressure summary stats
        pressure_drop = p[0] - p[-1]
        avg_gradient = pressure_drop / depth if depth > 0 else 0
        st.metric("Total Pressure Drop", f"{pressure_drop:.0f} psi", f"{avg_gradient:.3f} psi/ft")
    
    with col2:
        st.subheader("Temperature Profile")
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=temps, y=depths,mode='lines+markers', name='Temperature',line=dict(color='blue', width=2),hovertemplate='<b>%{x:.1f} °F</b> at %{y:.0f} ft'))
 
        fig_temp.update_layout(yaxis_title='Depth (ft)',xaxis_title='Temperature (°F)', yaxis=dict(autorange='reversed',title_font=dict(family= 'Times New Roman', color='black', size=20), tickfont=dict(color='black', size=20)),
                               xaxis=dict(title_font=dict(family= 'Times New Roman', color='black', size=20), tickfont=dict(color='black', size=20)), height=600,hovermode='x unified')
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Temperature summary stats
        temp_increase = temps[-1] - temps[0]
        geo_gradient = temp_increase / depth * 1000 if depth > 0 else 0
        st.metric("Temperature Increase", f"{temp_increase:.1f} °F", f"{geo_gradient:.1f} °F/1000ft")

with tab2:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Optimisation Parameters")
        # Optimization controls
        new_gor = st.slider('Gas Oil Ratio (SCF/STB)', 100, 3000, int(gor), 10,help="Adjust gas-oil ratio for sensitivity analysis")               
        new_thp = st.slider('Wellhead Pressure (psi)', 100, 5000, int(thp), 10, help="Adjust tubing head pressure")      
        new_diameter = st.slider('Tubing Size (inches)', MIN_TUBING_DIAM, MAX_TUBING_DIAM, float(diameter), 0.1, help="Change tubing internal diameter")

        # Optimization button
        if st.button("Update Analysis", help="Recalculate with new parameters"):
            with st.spinner("Recalculating performance..."):
                # Update parameters
                new_params = params.copy()
                new_params['gor'] = float(new_gor)
                new_params['thp'] = float(new_thp)
                new_params['diameter'] = float(new_diameter)
                
                # Recalculate VLP
                new_bhps = calculate_vlp(rate_range, new_params)
                new_optimal = find_intersection(rate_range, new_bhps, pwfs)
                
                # Store results
                st.session_state.nodal_data = {'params': new_params,'bhps': new_bhps,'optimal': new_optimal}
                st.success("Analysis updated successfully!")
    
    with col2:
        st.subheader("Nodal Analysis")
        fig_nodal = go.Figure()
        
        # Current VLP
        fig_nodal.add_trace(go.Scatter(x=rate_range, y=bhps,mode='lines', name='VLP (Current)',line=dict(color='blue', width=3),hovertemplate='%{y:.0f} psi at %{x:.0f} STB/d'))
 
        # Optimized VLP (if available) - Safe access
        nodal_data = st.session_state.get('nodal_data', {})
        if 'bhps' in nodal_data:
            fig_nodal.add_trace(go.Scatter(x=rate_range, y=nodal_data['bhps'], mode='lines', name='VLP (Optimized)',line=dict(color='green', width=3, dash='dot'),hovertemplate='%{y:.0f} psi at %{x:.0f} STB/d'))
 
        # IPR curve
        fig_nodal.add_trace(go.Scatter(x=rate_range, y=pwfs,mode='lines',name='IPR',line=dict(color='red', width=3),hovertemplate='%{y:.0f} psi at %{x:.0f} STB/d' ))
        
        # Optimal rate markers
        if optimal_rate:
            fig_nodal.add_vline( x=optimal_rate, line=dict(color="purple", width=2, dash="dash"), annotation_text=f"Optimal: {optimal_rate:.0f} STB/d", annotation_position="top left" )
        
        # Optimized rate marker (safe access)
        if 'optimal' in nodal_data and nodal_data['optimal']:
            fig_nodal.add_vline( x=nodal_data['optimal'], line=dict(color="orange", width=2, dash="dash"), annotation_text=f"Optimized: {nodal_data['optimal']:.0f} STB/d", annotation_position="bottom right" )
        fig_nodal.update_layout( xaxis_title='Liquid Rate (STB/d)', yaxis_title='Bottomhole Pressure (psi)', legend=dict(orientation="h", yanchor="bottom", y=1.02), 
                        xaxis=dict(title_font=dict(family='Times New Roman', color='black', size=20),tickfont=dict(family='Times New Roman', color='black', size=20)),
                        yaxis=dict( title_font=dict(family='Times New Roman', color='black', size=20), tickfont=dict(family='Times New Roman', color='black', size=20)), height=600, hovermode='x unified')
        st.plotly_chart(fig_nodal, use_container_width=True)
        
        # Performance summary
        if optimal_rate:
            gain = optimal_rate - oil_rate
            st.metric("Production Potential", f"{optimal_rate:.0f} STB/d", f"{gain:.0f} STB/d ({gain/oil_rate*100:.1f}%)" if gain > 0 else "No gain")
                    

with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Flow Pattern Distribution")
        
        # Flow pattern by depth
        fig_pattern = go.Figure()
        colors = {'Segregated': '#1f77b4','Intermittent': '#ff7f0e','Distributed': '#2ca02c','Transition': '#9467bd','Unknown': '#7f7f7f','Error': '#d62728'}
 
        # Group by pattern
        for pattern in sorted(set(patterns)):
            pattern_data = pattern_df[pattern_df['Flow Pattern'] == pattern]
            if not pattern_data.empty:
                fig_pattern.add_trace(go.Scatter(x=pattern_data['Pressure (psi)'],y=pattern_data['Depth (ft)'],mode='markers',name=pattern, marker=dict(color=colors.get(pattern.split(':')[0], '#7f7f7f'),
                size=8, opacity=0.8,line=dict(width=1, color='DarkSlateGrey')),hovertemplate='<b>%{text}</b><br>%{x:.0f} psi at %{y:.0f} ft',text=[pattern]*len(pattern_data) ))
     
        fig_pattern.update_layout(title='Flow Pattern Distribution',yaxis_title='Depth (ft)', xaxis_title='Pressure (psi)',yaxis=dict(autorange='reversed', title_font=dict(family='Times New Roman', color='black', size=20),
                tickfont=dict(family='Times New Roman', color='black', size=20)), xaxis=dict(title_font=dict(family='Times New Roman', color='black', size=20),
                tickfont=dict(family='Times New Roman', color='black', size=20)), height=700,legend_title="Flow Patterns")
        st.plotly_chart(fig_pattern, use_container_width=True)


    with col2:
        st.subheader("Flow Pattern Analysis")
        
        # Flow pattern table - Fixed with map instead of applymap
        st.dataframe(
            pattern_df.style.map(
                lambda x: 'background-color: #1f77b4; color: white' if 'Segregated' in str(x) else 
                         'background-color: #ff7f0e; color: white' if 'Intermittent' in str(x) else 
                         'background-color: #2ca02c; color: white' if 'Distributed' in str(x) else 
                         'background-color: #d62728; color: white' if 'Error' in str(x) else '',
                subset=['Flow Pattern']), height=400 )
        
        # Flow pattern summary
        pattern_counts = pattern_df['Flow Pattern'].value_counts()
        pattern_summary = pd.DataFrame({ 'Pattern': pattern_counts.index, 'Count': pattern_counts.values, 'Percentage': pattern_counts.values / len(pattern_df) * 100 })
            
        # Display summary
        st.dataframe(pattern_summary.style.format({'Percentage': '{:.1f}%'}).background_gradient(cmap='Blues'),
            height=300)
        
# Performance Summary

st.divider()
st.subheader("Well Performance Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Bottomhole Pressure", f"{p[-1]:.0f} psi", "BHP")
col2.metric("Optimal Rate", 
           f"{optimal_rate:.0f} STB/d" if optimal_rate else "N/A", 
           "Production potential" if optimal_rate else "")
col3.metric("Dominant Flow", 
           pattern_df['Flow Pattern'].value_counts().idxmax(), 
           "Most common pattern")
col4.metric("Efficiency", 
           f"{(oil_rate / optimal_rate * 100):.1f}%" if optimal_rate and optimal_rate > 0 else "N/A", 
           "Current/Optimal")
