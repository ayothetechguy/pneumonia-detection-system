"""
═══════════════════════════════════════════════════════════════
PNEUMONIA DETECTION SYSTEM - World-Class Medical Interface
═══════════════════════════════════════════════════════════════
AI-powered chest X-ray analysis with patient risk assessment
Trained Model Accuracy: 85.58%
Author: Ayoolumi Melehon
═══════════════════════════════════════════════════════════════
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import io
from download_model import download_model
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PERFECTED MEDICAL UI CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Base styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 15px !important;
    }
    
    /* Main background with medical imagery */
    .main {
        background: 
            linear-gradient(135deg, rgba(10, 25, 41, 0.97) 0%, rgba(13, 71, 161, 0.95) 50%, rgba(26, 35, 126, 0.97) 100%),
            url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 600"><defs><pattern id="medical-pattern" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1.5" fill="%2342a5f5" opacity="0.3"/><path d="M48,45 h4 v-8 h8 v4 h-8 v8 h-4 Z" fill="%2342a5f5" opacity="0.15"/></pattern></defs><rect width="1200" height="600" fill="url(%23medical-pattern)"/></svg>');
        background-attachment: fixed;
        background-size: cover;
        position: relative;
        min-height: 100vh;
    }
    
    /* Animated heartbeat line */
    .main::before {
        content: '';
        position: fixed;
        top: 40%;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(76, 175, 80, 0) 10%,
            rgba(76, 175, 80, 0.4) 30%, 
            rgba(76, 175, 80, 0.8) 50%, 
            rgba(76, 175, 80, 0.4) 70%, 
            rgba(76, 175, 80, 0) 90%,
            transparent 100%);
        animation: heartbeat-line 4s ease-in-out infinite;
        pointer-events: none;
        z-index: 1;
        filter: drop-shadow(0 0 10px rgba(76, 175, 80, 0.5));
    }
    
    @keyframes heartbeat-line {
        0% { transform: translateX(-100%) scaleY(1); opacity: 0; }
        5% { opacity: 0.6; }
        20% { transform: translateX(-50%) scaleY(2); }
        25% { transform: translateX(-40%) scaleY(0.5); }
        30% { transform: translateX(-30%) scaleY(2.5); }
        35% { transform: translateX(-20%) scaleY(1); }
        50% { transform: translateX(0%) scaleY(1); opacity: 0.8; }
        95% { opacity: 0.6; }
        100% { transform: translateX(100%) scaleY(1); opacity: 0; }
    }
    
    /* DNA helix background */
    .main::after {
        content: '';
        position: fixed;
        top: 0;
        right: 10%;
        width: 200px;
        height: 100%;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 400"><path d="M20,0 Q50,50 20,100 T20,200 T20,300 T20,400 M80,0 Q50,50 80,100 T80,200 T80,300 T80,400" stroke="%232196f3" stroke-width="2" fill="none" opacity="0.1"/><circle cx="20" cy="50" r="3" fill="%2342a5f5" opacity="0.2"/><circle cx="80" cy="50" r="3" fill="%2342a5f5" opacity="0.2"/><circle cx="20" cy="150" r="3" fill="%2342a5f5" opacity="0.2"/><circle cx="80" cy="150" r="3" fill="%2342a5f5" opacity="0.2"/><circle cx="20" cy="250" r="3" fill="%2342a5f5" opacity="0.2"/><circle cx="80" cy="250" r="3" fill="%2342a5f5" opacity="0.2"/><circle cx="20" cy="350" r="3" fill="%2342a5f5" opacity="0.2"/><circle cx="80" cy="350" r="3" fill="%2342a5f5" opacity="0.2"/></svg>');
        background-repeat: repeat-y;
        background-size: 100% auto;
        opacity: 0.3;
        pointer-events: none;
        animation: dna-float 20s linear infinite;
    }
    
    @keyframes dna-float {
        from { background-position: 0 0; }
        to { background-position: 0 400px; }
    }
    
    /* Floating medical icons */
    .medical-icon-float {
        position: fixed;
        font-size: 6rem;
        opacity: 0.03;
        animation: float-diagonal 30s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    .medical-icon-1 { top: 10%; left: 5%; animation-delay: 0s; }
    .medical-icon-2 { top: 60%; left: 80%; animation-delay: 5s; }
    .medical-icon-3 { top: 30%; right: 10%; animation-delay: 10s; }
    
    @keyframes float-diagonal {
        0%, 100% { transform: translate(0, 0) rotate(0deg); opacity: 0.03; }
        25% { transform: translate(30px, -30px) rotate(5deg); opacity: 0.05; }
        50% { transform: translate(0, -60px) rotate(-5deg); opacity: 0.04; }
        75% { transform: translate(-30px, -30px) rotate(3deg); opacity: 0.05; }
    }
    
    /* Premium glass header */
    .world-class-header {
        background: linear-gradient(135deg, 
            rgba(25, 118, 210, 0.98) 0%, 
            rgba(13, 71, 161, 0.98) 50%, 
            rgba(66, 165, 245, 0.98) 100%);
        backdrop-filter: blur(30px) saturate(150%);
        -webkit-backdrop-filter: blur(30px) saturate(150%);
        border: 2px solid rgba(255, 255, 255, 0.25);
        padding: 2.5rem;
        border-radius: 25px;
        text-align: center;
        box-shadow: 
            0 25px 70px rgba(13, 71, 161, 0.6),
            inset 0 2px 0 rgba(255, 255, 255, 0.4),
            0 0 80px rgba(33, 150, 243, 0.3);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        animation: header-glow 4s ease-in-out infinite;
    }
    
    @keyframes header-glow {
        0%, 100% { box-shadow: 0 25px 70px rgba(13, 71, 161, 0.6), inset 0 2px 0 rgba(255, 255, 255, 0.4), 0 0 80px rgba(33, 150, 243, 0.3); }
        50% { box-shadow: 0 25px 70px rgba(13, 71, 161, 0.8), inset 0 2px 0 rgba(255, 255, 255, 0.5), 0 0 100px rgba(33, 150, 243, 0.5); }
    }
    
    .world-class-header::before {
        content: '🫁';
        position: absolute;
        font-size: 18rem;
        opacity: 0.1;
        right: -6rem;
        top: -6rem;
        animation: lung-breathe 6s ease-in-out infinite;
        filter: drop-shadow(0 0 30px rgba(255, 255, 255, 0.4));
    }
    
    @keyframes lung-breathe {
        0%, 100% { transform: translateY(0) scale(1) rotate(0deg); }
        50% { transform: translateY(-30px) scale(1.05) rotate(5deg); }
    }
    
    .world-class-header::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(
            from 0deg,
            transparent 0deg,
            rgba(255, 255, 255, 0.15) 90deg,
            transparent 180deg,
            rgba(255, 255, 255, 0.15) 270deg,
            transparent 360deg
        );
        animation: header-shine 6s linear infinite;
    }
    
    @keyframes header-shine {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .world-class-header h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
        text-shadow: 
            2px 2px 4px rgba(0,0,0,0.4),
            0 0 25px rgba(255, 255, 255, 0.3),
            0 0 50px rgba(66, 165, 245, 0.5);
        letter-spacing: -1.5px;
        position: relative;
        z-index: 1;
    }
    
    .world-class-header p {
        color: #e3f2fd !important;
        font-size: 1.2rem !important;
        margin-top: 1rem !important;
        font-weight: 500 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    /* Glass morphism info banner */
    .premium-glass-banner {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.98) 0%, 
            rgba(232, 245, 233, 0.98) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 2px solid rgba(76, 175, 80, 0.4);
        padding: 1.8rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 
            0 15px 40px rgba(76, 175, 80, 0.25),
            inset 0 2px 0 rgba(255, 255, 255, 0.9);
        position: relative;
        overflow: hidden;
    }
    
    .premium-glass-banner::before {
        content: '✓';
        position: absolute;
        font-size: 15rem;
        color: rgba(76, 175, 80, 0.04);
        right: -2rem;
        top: -5rem;
        font-weight: 900;
        animation: check-pulse 3s ease-in-out infinite;
    }
    
    @keyframes check-pulse {
        0%, 100% { transform: scale(1); opacity: 0.04; }
        50% { transform: scale(1.05); opacity: 0.06; }
    }
    
    .premium-glass-banner h4 {
        color: #1b5e20 !important;
        margin: 0 0 0.8rem 0 !important;
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .premium-glass-banner p {
        color: #2e7d32 !important;
        font-size: 1rem !important;
        margin: 0 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar premium styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(13, 71, 161, 0.98) 0%, 
            rgba(21, 101, 192, 0.98) 100%);
        backdrop-filter: blur(30px);
        border-right: 4px solid rgba(66, 165, 245, 0.6);
        box-shadow: 5px 0 40px rgba(13, 71, 161, 0.4);
    }
    
    /* Medical cards */
    .world-class-card {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.98) 0%, 
            rgba(255, 255, 255, 0.95) 100%);
        backdrop-filter: blur(25px) saturate(150%);
        -webkit-backdrop-filter: blur(25px) saturate(150%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.18),
            inset 0 2px 0 rgba(255, 255, 255, 0.9),
            0 0 0 2px rgba(255, 255, 255, 0.4);
        border: 2px solid rgba(33, 150, 243, 0.25);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .world-class-card::before {
        content: '';
        position: absolute;
        top: -100%;
        left: -100%;
        width: 300%;
        height: 300%;
        background: conic-gradient(
            from 0deg at 50% 50%,
            rgba(33, 150, 243, 0) 0deg,
            rgba(33, 150, 243, 0.08) 90deg,
            rgba(33, 150, 243, 0) 180deg,
            rgba(33, 150, 243, 0.08) 270deg,
            rgba(33, 150, 243, 0) 360deg
        );
        animation: card-rotate 8s linear infinite;
        opacity: 0;
        transition: opacity 0.5s;
    }
    
    .world-class-card:hover::before {
        opacity: 1;
    }
    
    .world-class-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 30px 80px rgba(0, 0, 0, 0.25),
            inset 0 2px 0 rgba(255, 255, 255, 1),
            0 0 0 2px rgba(33, 150, 243, 0.5);
        border-color: rgba(33, 150, 243, 0.5);
    }
    
    @keyframes card-rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .world-class-card h4 {
        color: #0d47a1 !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
        font-weight: 800 !important;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        position: relative;
        z-index: 1;
    }
    
    .world-class-card p {
        color: #1565c0 !important;
        font-size: 1rem !important;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    
    /* Premium risk boxes */
    .world-class-risk-box {
        padding: 2.5rem;
        border-radius: 25px;
        margin: 2rem 0;
        font-size: 2rem !important;
        font-weight: 900;
        text-align: center;
        box-shadow: 0 25px 70px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
        border: 4px solid;
        animation: risk-pulse 3s ease-in-out infinite;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    @keyframes risk-pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .world-class-risk-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.35) 0%, transparent 70%);
        animation: risk-shimmer 5s ease-in-out infinite;
    }
    
    @keyframes risk-shimmer {
        0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.35; }
        50% { transform: translate(25%, 25%) scale(1.1); opacity: 0.6; }
    }
    
    .low-risk-final {
        background: linear-gradient(135deg, 
            rgba(129, 199, 132, 1) 0%, 
            rgba(165, 214, 167, 1) 100%);
        color: #1b5e20;
        border-color: #4caf50;
        box-shadow: 
            0 25px 70px rgba(76, 175, 80, 0.5),
            inset 0 3px 0 rgba(255, 255, 255, 0.6),
            0 0 50px rgba(76, 175, 80, 0.4);
    }
    
    .medium-risk-final {
        background: linear-gradient(135deg, 
            rgba(255, 249, 196, 1) 0%, 
            rgba(255, 245, 157, 1) 100%);
        color: #f57f17;
        border-color: #fbc02d;
        box-shadow: 
            0 25px 70px rgba(251, 192, 45, 0.5),
            inset 0 3px 0 rgba(255, 255, 255, 0.6),
            0 0 50px rgba(251, 192, 45, 0.4);
    }
    
    .high-risk-final {
        background: linear-gradient(135deg, 
            rgba(239, 154, 154, 1) 0%, 
            rgba(229, 115, 115, 1) 100%);
        color: #b71c1c;
        border-color: #f44336;
        box-shadow: 
            0 25px 70px rgba(244, 67, 54, 0.5),
            inset 0 3px 0 rgba(255, 255, 255, 0.6),
            0 0 50px rgba(244, 67, 54, 0.4);
    }
    
    /* Enhanced metrics */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, 
            rgba(227, 242, 253, 1) 0%, 
            rgba(255, 255, 255, 1) 100%);
        backdrop-filter: blur(15px);
        padding: 1.8rem;
        border-radius: 20px;
        border: 3px solid rgba(33, 150, 243, 0.4);
        box-shadow: 
            0 15px 40px rgba(33, 150, 243, 0.25),
            inset 0 2px 0 rgba(255, 255, 255, 1);
        transition: all 0.4s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-6px) scale(1.03);
        box-shadow: 
            0 25px 60px rgba(33, 150, 243, 0.35),
            inset 0 2px 0 rgba(255, 255, 255, 1);
        border-color: rgba(33, 150, 243, 0.6);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 50%, #64b5f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(2px 2px 4px rgba(25, 118, 210, 0.4));
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        color: #0d47a1 !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* PERFECTED TABS - PROFESSIONAL & VISIBLE */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background: linear-gradient(135deg, 
            rgba(240, 248, 255, 0.95) 0%, 
            rgba(255, 255, 255, 0.95) 100%);
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.12);
    }
    
    /* INACTIVE TABS - BEAUTIFUL GRADIENT WITH DARK TEXT */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        padding: 1.3rem 2.5rem !important;
        background: linear-gradient(135deg, 
            rgba(187, 222, 251, 0.9) 0%, 
            rgba(144, 202, 249, 0.9) 100%) !important;
        border-radius: 15px;
        color: #0d47a1 !important;
        border: 3px solid rgba(25, 118, 210, 0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 18px rgba(33, 150, 243, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255,255,255,0.5), 
            transparent);
        transition: left 0.6s;
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        left: 100%;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, 
            rgba(100, 181, 246, 0.95) 0%, 
            rgba(66, 165, 245, 0.95) 100%) !important;
        border-color: #1976d2 !important;
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 12px 30px rgba(25, 118, 210, 0.35);
        color: #01579b !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, 
            #1976d2 0%, 
            #42a5f5 50%, 
            #64b5f6 100%) !important;
        color: white !important;
        border-color: #0d47a1 !important;
        box-shadow: 
            0 12px 40px rgba(25, 118, 210, 0.6),
            inset 0 2px 0 rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-3px);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h1 { font-size: 2.8rem !important; letter-spacing: -1px; }
    h2 { font-size: 2.2rem !important; letter-spacing: -0.5px; }
    h3 { font-size: 1.8rem !important; }
    h4 { font-size: 1.4rem !important; }
    
    /* Paragraph text */
    p, li, span {
        color: #e3f2fd !important;
        font-size: 1rem !important;
    }
    
    /* Premium buttons */
    .stButton button {
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        padding: 1.2rem 2.8rem !important;
        border-radius: 50px;
        border: none;
        box-shadow: 
            0 12px 35px rgba(25, 118, 210, 0.5),
            inset 0 2px 0 rgba(255, 255, 255, 0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.4);
        transform: translate(-50%, -50%);
        transition: width 0.8s, height 0.8s;
    }
    
    .stButton button:hover::before {
        width: 350px;
        height: 350px;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
        box-shadow: 
            0 18px 45px rgba(13, 71, 161, 0.6),
            inset 0 2px 0 rgba(255, 255, 255, 0.5);
        transform: translateY(-4px) scale(1.05);
    }
    
    /* Enhanced input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        font-size: 1.1rem !important;
        padding: 1rem !important;
        border: 3px solid rgba(187, 222, 251, 0.6) !important;
        border-radius: 15px !important;
        background: rgba(255, 255, 255, 0.98) !important;
        color: #0d47a1 !important;
        font-weight: 600 !important;
        backdrop-filter: blur(15px);
        transition: all 0.4s ease;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #2196f3 !important;
        box-shadow: 
            0 0 0 4px rgba(33, 150, 243, 0.2),
            0 12px 30px rgba(33, 150, 243, 0.25);
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 1) !important;
    }
    
    /* Labels */
    label {
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin-bottom: 0.6rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced checkboxes */
    .stCheckbox {
        font-size: 1.1rem !important;
        background: rgba(255, 255, 255, 0.12);
        padding: 0.8rem;
        border-radius: 10px;
        transition: all 0.3s;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .stCheckbox:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
    }
    
    .stCheckbox label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    /* Alert boxes */
    /* Alert boxes */
    /* Alert boxes */
    /* Alert boxes */
    .stAlert {
        font-size: 1.1rem !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        border-left: 5px solid;
        backdrop-filter: blur(15px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.12);
        font-weight: 600 !important;
    }

    /* Make alert/info text BLACK for readability */
    .stAlert p, .stAlert div, .stAlert span {
        color: #000000 !important;
    }

    /* Make main content headers BLACK for readability */
    .main h2, .main h3, .main h4 {
        color: #000000 !important;
        text-shadow: none !important;
    }

    /* Make alert/info text BLACK for readability */

    /* Make main content headers BLACK for readability */
    .main h2, .main h3, .main h4 {
        color: #0d47a1 !important;
        text-shadow: none !important;
    }
    .stAlert p, .stAlert div, .stAlert span {
        color: #000000 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4caf50 0%, #81c784 50%, #a5d6a7 100%);
        height: 25px;
        border-radius: 15px;
        box-shadow: 
            0 6px 18px rgba(76, 175, 80, 0.4),
            inset 0 2px 0 rgba(255, 255, 255, 0.6);
        animation: progress-glow 2s ease-in-out infinite;
    }
    
    @keyframes progress-glow {
        0%, 100% { filter: brightness(1) saturate(1); }
        50% { filter: brightness(1.2) saturate(1.3); }
    }
    
    /* File uploader */
    .stFileUploader {
        background: linear-gradient(135deg, 
            rgba(227, 242, 253, 0.98) 0%, 
            rgba(255, 255, 255, 0.98) 100%);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 20px;
        border: 4px dashed rgba(33, 150, 243, 0.6);
        transition: all 0.4s ease;
        box-shadow: 0 12px 35px rgba(0,0,0,0.12);
    }
    
    .stFileUploader:hover {
        border-color: #2196f3;
        background: linear-gradient(135deg, 
            rgba(187, 222, 251, 0.98) 0%, 
            rgba(227, 242, 253, 0.98) 100%);
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 18px 45px rgba(33, 150, 243, 0.25);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, 
            rgba(227, 242, 253, 0.98) 0%, 
            rgba(255, 255, 255, 0.98) 100%);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.4rem !important;
        color: #0d47a1 !important;
        border: 3px solid rgba(33, 150, 243, 0.3);
        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
        transition: all 0.4s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, 
            rgba(187, 222, 251, 0.98) 0%, 
            rgba(227, 242, 253, 0.98) 100%);
        box-shadow: 0 10px 25px rgba(33, 150, 243, 0.2);
        transform: translateY(-2px);
    }
    
    /* Disclaimer box with BLACK text */
    .disclaimer-box {
        background: linear-gradient(135deg, rgba(255, 249, 196, 1) 0%, rgba(255, 245, 157, 1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #f57f17;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .disclaimer-box p {
        color: #000000 !important;
        margin: 0 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    .disclaimer-box strong {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    /* World-class footer */
    .world-class-footer {
        background: linear-gradient(135deg, 
            rgba(13, 71, 161, 0.98) 0%, 
            rgba(25, 118, 210, 0.98) 100%);
        backdrop-filter: blur(30px);
        border: 2px solid rgba(255, 255, 255, 0.25);
        color: white;
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 
            0 25px 70px rgba(13, 71, 161, 0.5),
            inset 0 2px 0 rgba(255, 255, 255, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .world-class-footer::before {
        content: '';
        position: absolute;
        top: -100%;
        left: -100%;
        width: 300%;
        height: 300%;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
        animation: footer-shimmer 10s ease-in-out infinite;
    }
    
    @keyframes footer-shimmer {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(15%, 15%); }
    }
    
    .world-class-footer h3 {
        color: white !important;
        font-size: 2rem !important;
        margin: 0 0 0.8rem 0 !important;
    }
    
    .world-class-footer p {
        color: #e3f2fd !important;
        font-size: 1rem !important;
        margin: 0.4rem 0 !important;
    }
</style>

<!-- Floating medical icons -->
<div class="medical-icon-float medical-icon-1">🫀</div>
<div class="medical-icon-float medical-icon-2">🧬</div>
<div class="medical-icon-float medical-icon-3">⚕️</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MODEL LOADING - UPDATED FOR GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load model with Google Drive download support"""
    
    # Download model from Google Drive if not present
    if not download_model():
        st.error("⚠️ Failed to download model from cloud storage. Please refresh the page.")
        return None, torch.device('cpu')
    
    device = torch.device('cpu')
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    
    try:
        # Use relative path that works on Streamlit Cloud
        model_path = 'models/best_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ═══════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def calculate_clinical_risk(age, has_fever, has_cough, has_breathing_difficulty, 
                           is_smoker, has_chronic_condition, symptom_days):
    risk_score = 0
    if age > 65: risk_score += 25
    elif age > 45: risk_score += 15
    elif age < 5: risk_score += 20
    if has_fever: risk_score += 15
    if has_cough: risk_score += 10
    if has_breathing_difficulty: risk_score += 20
    if symptom_days > 7: risk_score += 15
    elif symptom_days > 3: risk_score += 10
    if is_smoker: risk_score += 10
    if has_chronic_condition: risk_score += 15
    return min(risk_score, 100)

def get_risk_category(risk_score):
    if risk_score < 30: return "LOW", "🟢"
    elif risk_score < 60: return "MEDIUM", "🟡"
    else: return "HIGH", "🔴"

def predict_xray(image, model, device):
    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        class_names = ['Normal', 'Pneumonia']
        prediction = class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        normal_prob = probabilities[0][0].item() * 100
        pneumonia_prob = probabilities[0][1].item() * 100
        return prediction, confidence_score, normal_prob, pneumonia_prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None, None

# ═══════════════════════════════════════════════════════════════
# PDF REPORT GENERATION
# ═══════════════════════════════════════════════════════════════

def create_risk_gauge_chart(risk_score, risk_category):
    """Create a colorful risk gauge chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create the gauge
    colors_map = ['#81c784', '#fff59d', '#ef9a9a']
    bounds = [0, 30, 60, 100]
    
    # Draw the gauge background
    for i in range(len(colors_map)):
        ax.barh(0, bounds[i+1] - bounds[i], left=bounds[i], height=0.3, 
                color=colors_map[i], alpha=0.7, edgecolor='white', linewidth=2)
    
    # Draw the risk pointer
    ax.plot([risk_score, risk_score], [-0.2, 0.5], 'k-', linewidth=4)
    ax.plot(risk_score, 0.5, 'ko', markersize=15)
    
    # Labels
    ax.text(15, -0.5, 'LOW\n0-30', ha='center', fontsize=12, fontweight='bold', color='#1b5e20')
    ax.text(45, -0.5, 'MEDIUM\n30-60', ha='center', fontsize=12, fontweight='bold', color='#f57f17')
    ax.text(80, -0.5, 'HIGH\n60-100', ha='center', fontsize=12, fontweight='bold', color='#b71c1c')
    
    # Risk score text
    ax.text(risk_score, 0.8, f'{risk_score}', ha='center', fontsize=24, fontweight='bold', color='#0d47a1')
    ax.text(risk_score, 1.2, f'{risk_category} RISK', ha='center', fontsize=14, fontweight='bold', color='#1565c0')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1.5)
    ax.axis('off')
    
    # Save to bytes
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return buf

def create_symptoms_chart(has_fever, has_cough, has_breathing_difficulty, is_smoker, has_chronic_condition):
    """Create a colorful symptoms presence chart"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    symptoms = ['Fever', 'Cough', 'Dyspnea', 'Smoking\nHistory', 'Chronic\nDisease']
    values = [has_fever, has_cough, has_breathing_difficulty, is_smoker, has_chronic_condition]
    colors_list = ['#ef5350' if v else '#e0e0e0' for v in values]
    
    bars = ax.barh(symptoms, [1]*5, color=colors_list, edgecolor='white', linewidth=2)
    
    # Add checkmarks or X marks
    for i, (symptom, value) in enumerate(zip(symptoms, values)):
        if value:
            ax.text(0.5, i, '✓ PRESENT', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
        else:
            ax.text(0.5, i, '✗ ABSENT', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='#757575')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Clinical Indicators', fontsize=12, fontweight='bold')
    ax.set_title('Symptom Profile', fontsize=16, fontweight='bold', color='#0d47a1', pad=20)
    ax.set_xticks([])
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save to bytes
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return buf

def generate_pdf_report(patient_name, age, gender, has_fever, has_cough, has_breathing_difficulty,
                        is_smoker, has_chronic_condition, symptom_days, risk_score, risk_category,
                        prediction=None, confidence=None, uploaded_image=None):
    """Generate a colorful, graphical PDF report"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50,
                           topMargin=50, bottomMargin=50)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0d47a1'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1976d2'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        borderWidth=2,
        borderColor=colors.HexColor('#42a5f5'),
        borderPadding=10,
        backColor=colors.HexColor('#e3f2fd')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=colors.HexColor('#263238'),
        spaceAfter=10,
        leading=16
    )
    
    # Header with colored background
    header_data = [[Paragraph('<b>🫁 PNEUMONIA DETECTION SYSTEM</b><br/>Medical Analysis Report', title_style)]]
    header_table = Table(header_data, colWidths=[7*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 18),
        ('TOPPADDING', (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ('GRID', (0, 0), (-1, -1), 2, colors.HexColor('#0d47a1')),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 20))
    
    # Report metadata
    report_time = datetime.now().strftime('%B %d, %Y at %H:%M:%S')
    report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    meta_data = [
        ['Report Generated:', report_time],
        ['Report ID:', report_id],
        ['Model Accuracy:', '85.58%']
    ]
    meta_table = Table(meta_data, colWidths=[2*inch, 4.5*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bbdefb')),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 20))
    
    # Patient Demographics Section
    elements.append(Paragraph('👤 PATIENT DEMOGRAPHICS', heading_style))
    
    patient_data = [
        ['Full Name:', patient_name],
        ['Age:', f'{age} years'],
        ['Gender:', gender],
        ['Assessment Date:', datetime.now().strftime('%Y-%m-%d')]
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4.5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#0d47a1')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#90caf9')),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 20))
    
    # Clinical Presentation Section
    elements.append(Paragraph('🩺 CLINICAL PRESENTATION', heading_style))
    
    clinical_data = [
        ['Pyrexia (Fever):', '✓ Present' if has_fever else '✗ Absent'],
        ['Persistent Cough:', '✓ Present' if has_cough else '✗ Absent'],
        ['Dyspnea:', '✓ Present' if has_breathing_difficulty else '✗ Absent'],
        ['Symptom Duration:', f'{symptom_days} days']
    ]
    clinical_table = Table(clinical_data, colWidths=[2*inch, 4.5*inch])
    clinical_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fff9c4')),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#f57f17')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#fff59d')),
    ]))
    elements.append(clinical_table)
    elements.append(Spacer(1, 20))
    
    # Medical History Section
    elements.append(Paragraph('⚠️ MEDICAL HISTORY', heading_style))
    
    history_data = [
        ['Tobacco Use:', '✓ Positive' if is_smoker else '✗ Negative'],
        ['Chronic Pulmonary Disease:', '✓ Positive' if has_chronic_condition else '✗ Negative']
    ]
    history_table = Table(history_data, colWidths=[2*inch, 4.5*inch])
    history_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ffccbc')),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#d84315')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ffab91')),
    ]))
    elements.append(history_table)
    elements.append(Spacer(1, 30))
    
    # Risk Assessment with Gauge Chart
    elements.append(Paragraph('📊 CLINICAL RISK ASSESSMENT', heading_style))
    
    # Create and add risk gauge chart
    risk_chart_buf = create_risk_gauge_chart(risk_score, risk_category)
    risk_chart_img = RLImage(risk_chart_buf, width=6*inch, height=3*inch)
    elements.append(risk_chart_img)
    elements.append(Spacer(1, 10))
    
    # Risk recommendation based on category
    if risk_category == "LOW":
        recommendation = "🏠 Home monitoring advised. Seek medical evaluation if symptoms worsen."
        rec_color = colors.HexColor('#4caf50')
    elif risk_category == "MEDIUM":
        recommendation = "🏥 Outpatient evaluation recommended. Consider diagnostic imaging."
        rec_color = colors.HexColor('#fbc02d')
    else:
        recommendation = "🚨 Immediate medical attention required. Emergency department assessment indicated."
        rec_color = colors.HexColor('#f44336')
    
    rec_data = [[Paragraph(f'<b>Clinical Recommendation:</b> {recommendation}', body_style)]]
    rec_table = Table(rec_data, colWidths=[6.5*inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), rec_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
    ]))
    elements.append(rec_table)
    elements.append(Spacer(1, 20))
    
    # Symptoms Chart
    symptoms_chart_buf = create_symptoms_chart(has_fever, has_cough, has_breathing_difficulty, 
                                                is_smoker, has_chronic_condition)
    symptoms_chart_img = RLImage(symptoms_chart_buf, width=6*inch, height=3.5*inch)
    elements.append(symptoms_chart_img)
    elements.append(Spacer(1, 20))
    
    # X-Ray Analysis Section (if available)
    if prediction:
        elements.append(PageBreak())
        elements.append(Paragraph('🔬 AI-POWERED RADIOGRAPHIC ANALYSIS', heading_style))
        
        xray_data = [
            ['AI Interpretation:', prediction.upper()],
            ['Confidence Level:', f'{confidence:.2f}%'],
            ['Model Architecture:', 'ResNet-18 Deep Learning'],
            ['Training Accuracy:', '85.58%']
        ]
        xray_table = Table(xray_data, colWidths=[2*inch, 4.5*inch])
        
        # Color based on prediction
        if prediction == "Normal":
            bg_color = colors.HexColor('#c8e6c9')
            text_color = colors.HexColor('#1b5e20')
        else:
            bg_color = colors.HexColor('#ffcdd2')
            text_color = colors.HexColor('#b71c1c')
        
        xray_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e1f5fe')),
            ('BACKGROUND', (1, 0), (1, 0), bg_color),
            ('TEXTCOLOR', (1, 0), (1, 0), text_color),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#01579b')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#81d4fa')),
        ]))
        elements.append(xray_table)
        elements.append(Spacer(1, 20))
    
    # Disclaimer
    elements.append(Spacer(1, 30))
    disclaimer_text = '''
    <b>⚕️ MEDICAL DISCLAIMER:</b><br/>
    This report is generated by an AI-assisted clinical decision support system. 
    All diagnostic findings and treatment recommendations must be reviewed and validated 
    by a qualified, licensed healthcare professional. This technology is intended to 
    augment, not replace, clinical judgment. The system is for research and educational 
    purposes only and is not FDA approved for clinical diagnostic use.
    '''
    disclaimer_para = Paragraph(disclaimer_text, body_style)
    
    disclaimer_data = [[disclaimer_para]]
    disclaimer_table = Table(disclaimer_data, colWidths=[6.5*inch])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff9c4')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('GRID', (0, 0), (-1, -1), 2, colors.HexColor('#f57f17')),
    ]))
    elements.append(disclaimer_table)
    
    # Footer
    elements.append(Spacer(1, 20))
    footer_text = '''
    <b>Pneumonia Detection System v1.0</b><br/>
    Developed by Ayoolumi Melehon | Clinical AI Technology<br/>
    © 2024 All Rights Reserved
    '''
    footer_para = Paragraph(footer_text, ParagraphStyle('Footer', parent=body_style, 
                                                        alignment=TA_CENTER, fontSize=9, 
                                                        textColor=colors.HexColor('#757575')))
    elements.append(footer_para)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return buffer

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

def main():
    # World-class Header
    st.markdown("""
    <div class="world-class-header">
        <h1>🫁 Pneumonia Detection System</h1>
        <p>AI-Powered Clinical Decision Support | World-Class Healthcare Technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Premium Glass Banner
    st.markdown("""
    <div class="premium-glass-banner">
        <h4>📊 Advanced Diagnostic Intelligence Platform</h4>
        <p>Model Accuracy: 85.58% | Deep Learning Architecture | Real-Time Analysis | Clinical-Grade AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, device = load_model()
    if model is None:
        st.error("⚠️ Model initialization failed.")
        return
    
    # Sidebar
    st.sidebar.markdown("### 👤 PATIENT DEMOGRAPHICS")
    patient_name = st.sidebar.text_input("Full Name", placeholder="Enter patient's full name")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🩺 CLINICAL SYMPTOMS")
    has_fever = st.sidebar.checkbox("🌡️ Fever")
    has_cough = st.sidebar.checkbox("😷 Cough")
    has_breathing_difficulty = st.sidebar.checkbox("😮‍💨 Dyspnea")
    symptom_days = st.sidebar.slider("Duration (days)", 0, 30, 3)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ MEDICAL HISTORY")
    is_smoker = st.sidebar.checkbox("🚬 Smoking History")
    has_chronic_condition = st.sidebar.checkbox("🫁 Chronic Lung Disease")
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Risk Assessment", "🔬 X-Ray Analysis", "📄 Medical Report"])
    
    # TAB 1
    with tab1:
        st.markdown("## 📊 Clinical Risk Assessment")
        
        if patient_name:
            risk_score = calculate_clinical_risk(
                age, has_fever, has_cough, has_breathing_difficulty,
                is_smoker, has_chronic_condition, symptom_days
            )
            risk_category, risk_icon = get_risk_category(risk_score)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{risk_score}/100")
            with col2:
                st.metric("Age", f"{age} years")
            with col3:
                st.metric("Duration", f"{symptom_days} days")
            
            st.markdown("#### Risk Level Indicator")
            st.progress(risk_score / 100)
            
            if risk_category == "LOW":
                css_class = "low-risk-final"
                recommendation = "🏠 Home monitoring advised. Seek evaluation if symptoms worsen."
            elif risk_category == "MEDIUM":
                css_class = "medium-risk-final"
                recommendation = "🏥 Outpatient evaluation recommended. Consider diagnostic imaging."
            else:
                css_class = "high-risk-final"
                recommendation = "🚨 Immediate medical attention required. ED assessment indicated."
            
            st.markdown(f'<div class="world-class-risk-box {css_class}">{risk_icon} {risk_category} RISK</div>', 
                       unsafe_allow_html=True)
            
            st.info(f"**Recommendation:** {recommendation}")
            
            with st.expander("📋 Detailed Risk Analysis"):
                st.markdown("### Contributing Factors:")
                factors = []
                if age > 65: factors.append("Advanced age (>65)")
                elif age < 5: factors.append("Pediatric (<5)")
                if has_fever: factors.append("Fever present")
                if has_cough: factors.append("Persistent cough")
                if has_breathing_difficulty: factors.append("Dyspnea")
                if symptom_days > 7: factors.append("Prolonged symptoms (>7 days)")
                if is_smoker: factors.append("Smoking history")
                if has_chronic_condition: factors.append("Chronic lung disease")
                
                if factors:
                    for factor in factors:
                        st.markdown(f"• {factor}")
                else:
                    st.success("✓ Minimal risk factors")
        else:
            st.info("👈 Enter patient information to begin assessment")
    
    # TAB 2
    with tab2:
        st.markdown("## 🔬 AI-Powered X-Ray Analysis")
        
        st.markdown("""
        <div class="world-class-card">
            <h4>📸 Upload Chest X-Ray</h4>
            <p>Upload frontal chest radiograph for AI diagnostic analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select X-ray image (JPEG/PNG)",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📷 Patient X-Ray")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("### 🤖 AI Analysis")
                
                with st.spinner("🔍 Analyzing..."):
                    prediction, confidence, normal_prob, pneumonia_prob = predict_xray(image, model, device)
                
                if prediction:
                    if prediction == "Normal":
                        st.success("✅ **NORMAL**")
                    else:
                        st.error("⚠️ **PNEUMONIA DETECTED**")
                    
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    st.markdown("#### Probabilities")
                    prob_data = pd.DataFrame({
                        'Class': ['Normal', 'Pneumonia'],
                        'Probability (%)': [normal_prob, pneumonia_prob]
                    })
                    st.bar_chart(prob_data.set_index('Class'))
                    
                    st.markdown("---")
                    st.markdown("### 🩺 Interpretation")
                    
                    if prediction == "Pneumonia":
                        if confidence > 90:
                            st.warning("High confidence detection. Immediate clinical correlation recommended.")
                        elif confidence > 70:
                            st.warning("Moderate confidence. Additional diagnostic workup advised.")
                        else:
                            st.info("Possible finding. Further evaluation recommended.")
                    else:
                        if confidence > 90:
                            st.success("High confidence normal. Low probability of pneumonia.")
                        else:
                            st.info("Appears normal. Clinical correlation recommended if symptomatic.")
                    
                    st.caption("⚕️ *AI Decision Support Tool. Requires validation by licensed healthcare professional.*")
        else:
            st.info("📤 Upload chest X-ray to begin analysis")
    
    # TAB 3 - ENHANCED WITH PDF DOWNLOAD
    with tab3:
        st.markdown("## 📄 Colorful Medical Report")
        
        if patient_name:
            # Calculate risk if not already done
            risk_score = calculate_clinical_risk(
                age, has_fever, has_cough, has_breathing_difficulty,
                is_smoker, has_chronic_condition, symptom_days
            )
            risk_category, risk_icon = get_risk_category(risk_score)
            
            # Get prediction if X-ray was uploaded
            prediction = None
            confidence = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                prediction, confidence, _, _ = predict_xray(image, model, device)
            
            # Display preview
            st.markdown(f"""
            ### 📋 Report Preview
            **Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}
            
            ---
            
            #### 👤 Patient: {patient_name}
            #### 📊 Risk Level: {risk_icon} **{risk_category}** ({risk_score}/100)
            #### 🔬 AI Analysis: {'**' + prediction.upper() + '**' if prediction else 'Not performed'}
            """)
            
            st.markdown("---")
            
            # Generate PDF button
            st.markdown("### 📥 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Generate PDF
                pdf_buffer = generate_pdf_report(
                    patient_name, age, gender, has_fever, has_cough, has_breathing_difficulty,
                    is_smoker, has_chronic_condition, symptom_days, risk_score, risk_category,
                    prediction, confidence, uploaded_file
                )
                
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"medical_report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with col2:
                # Text report for backup
                report_text = f"""
═══════════════════════════════════════════════════════════════
PNEUMONIA DETECTION SYSTEM - MEDICAL REPORT
═══════════════════════════════════════════════════════════════

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT INFORMATION
Name: {patient_name}
Age: {age} years
Gender: {gender}

CLINICAL PRESENTATION
Fever: {'Yes' if has_fever else 'No'}
Cough: {'Yes' if has_cough else 'No'}
Dyspnea: {'Yes' if has_breathing_difficulty else 'No'}
Duration: {symptom_days} days

RISK ASSESSMENT
Risk Score: {risk_score}/100
Category: {risk_category}

{'X-RAY ANALYSIS\nPrediction: ' + prediction.upper() + '\nConfidence: ' + f'{confidence:.2f}%' if prediction else ''}

AI-assisted analysis. Requires professional validation.
═══════════════════════════════════════════════════════════════
                """
                
                st.download_button(
                    label="📝 Download Text Report",
                    data=report_text,
                    file_name=f"report_text_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Disclaimer
            st.markdown("""
            <div class="disclaimer-box">
                <p><strong>⚕️ Disclaimer:</strong> AI-assisted analysis. Requires validation by licensed medical professional.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visual report preview
            st.markdown("### 📊 Visual Report Preview")
            
            # Show risk gauge
            risk_chart_buf = create_risk_gauge_chart(risk_score, risk_category)
            st.image(risk_chart_buf, use_container_width=True)
            
            # Show symptoms chart
            symptoms_chart_buf = create_symptoms_chart(has_fever, has_cough, has_breathing_difficulty, 
                                                        is_smoker, has_chronic_condition)
            st.image(symptoms_chart_buf, use_container_width=True)
            
        else:
            st.info("👈 Enter patient information to generate colorful medical report")
    
    # World-class Footer
    st.markdown("""
    <div class="world-class-footer">
        <h3>🫁 Pneumonia Detection System</h3>
        <p>AI Model Accuracy: 85.58% | ResNet-18 Architecture</p>
        <p>Developed by Ayoolumi Melehon | Clinical Decision Support Technology</p>
        <p>Research & Educational Platform | Not FDA Approved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
