# src/app.py
import streamlit as st
from .src.ui.option_visualizer import OptionVisualizerApp

def main():
    app = OptionVisualizerApp()
    app.run()

if __name__ == "__main__":
    main()