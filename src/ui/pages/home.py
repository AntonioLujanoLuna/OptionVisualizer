# src/ui/pages/home.py

import streamlit as st
from datetime import datetime

def render_page():
    """Render the educational home page."""
    st.markdown(
        """
        # Learn Options Trading Interactively
        
        Welcome to your interactive options learning journey! This tool helps you understand:
        
        - Core options concepts through hands-on examples
        - How different pricing models work and compare
        - Option strategies and when to use them
        - The Greeks and risk management
        """
    )
    
    # Display concept of the day
    st.subheader("Today's Learning Concept")
    concepts = {
        0: ("Delta", "Learn how delta measures an option's directional exposure"),
        1: ("Time Decay", "Understand how options lose value as expiration approaches"),
        2: ("Implied Volatility", "Explore how market prices imply future volatility"),
        3: ("Put-Call Parity", "Discover the fundamental relationship between puts and calls"),
        4: ("Option Strategies", "Study common option combinations and their uses")
    }
    day_of_year = datetime.now().timetuple().tm_yday
    concept, description = concepts[day_of_year % len(concepts)]
    
    st.info(f"📚 **{concept}**: {description}")
    if st.button("Explore This Concept"):
        st.session_state.selected_concept = concept
    
    # Quick start sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Start Learning")
        st.markdown("""
            Choose your path:
            
            1. **Beginner Track**
               - Options Basics
               - Understanding Calls and Puts
               - Basic Strategies
            
            2. **Intermediate Track**
               - The Greeks
               - Volatility and Pricing
               - Spreads and Combinations
            
            3. **Advanced Track**
               - Advanced Strategies
               - Pricing Models Deep Dive
               - Volatility Trading
        """)
        
        track = st.selectbox(
            "Select Your Track",
            ["Beginner", "Intermediate", "Advanced"]
        )
        if st.button("Start Learning"):
            st.session_state.selected_track = track
    
    with col2:
        st.subheader("Interactive Tools")
        st.markdown("""
            Explore and experiment:
            
            - **Model Explorer**: Compare different pricing models
            - **Strategy Builder**: Build and analyze strategies
            - **Greeks Lab**: Interactive Greek exploration
            - **Volatility Tools**: Understand implied volatility
        """)
        
        tool = st.selectbox(
            "Select a Tool",
            ["Model Explorer", "Strategy Builder", "Greeks Lab", "Volatility Tools"]
        )
        if st.button("Launch Tool"):
            st.session_state.selected_tool = tool
    
    # Learning progress
    st.subheader("Your Learning Progress")
    concepts_learned = st.session_state.get('concepts_learned', [])
    
    progress = len(concepts_learned) / 20  # Example: 20 total concepts
    st.progress(progress)
    st.markdown(f"**{len(concepts_learned)}** of 20 core concepts mastered")
    
    if concepts_learned:
        st.markdown("Recently mastered concepts:")
        for concept in concepts_learned[-3:]:
            st.markdown(f"✅ {concept}")
    
    # Daily challenge
    st.subheader("Daily Challenge")
    challenges = {
        0: "Calculate the break-even point for a call option",
        1: "Explain how gamma changes as price moves",
        2: "Build a collar strategy",
        3: "Compare Black-Scholes vs Binomial prices",
        4: "Analyze a straddle's volatility exposure"
    }
    challenge = challenges[day_of_year % len(challenges)]
    
    st.info(f"🎯 **Today's Challenge**: {challenge}")
    if st.button("Try Challenge"):
        st.session_state.selected_challenge = challenge
    
    # Learning tips
    st.markdown("""
        ---
        ### 💡 Learning Tips
        
        - Start with the fundamentals before advancing
        - Use interactive tools to experiment with concepts
        - Complete daily challenges to reinforce learning
        - Review mastered concepts periodically
    """)