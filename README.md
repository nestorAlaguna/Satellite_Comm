# Optical Satellite Communications Digital Twin

A comprehensive Model-Based Systems Engineering (MBSE) simulation of optical satellite communication systems, integrating orbital mechanics, link budget analysis, and machine learning for predictive maintenance.

## Project Overview

This project implements a digital twin for optical satellite communication systems, demonstrating:

- **Model-Based Systems Engineering (MBSE)** principles using Python
- **Optical communication physics** with realistic link budget calculations
- **Orbital mechanics** integration using real satellite data (TLEs)
- **Machine Learning** for signal fade prediction and system optimization
- **Professional data visualization** and system performance analysis

## System Architecture

The digital twin models complete satellite communication chain:

- **Satellite Blocks**: Optical communication payloads with transmit power, antenna gain, and orbital parameters
- **Ground Station Blocks**: Optical receivers with telescope apertures and location constraints
- **Optical Link Model**: Physics-based link budget calculation including free-space path loss, atmospheric effects, and bit error rate analysis
- **Orbital Simulation**: Real-time satellite position tracking and visibility analysis
- **ML Prediction**: Adaptive fade prediction using historical telemetry data

## Key Features
- **MBSE Implementation**

    System Blocks: Satellite, GroundStation, OpticalLink classes with validated attributes

    Requirements Management: Formal requirement definition and verification

    Parametric Models: Physics-based link budget equations

    Behavioral Models: Ground station tracking logic and system states

- **Optical Communication Physics**

    Free-space optical path loss calculations

    Atmospheric attenuation modeling (clear, cloudy, turbulent conditions)

    Receiver sensitivity and bit error rate analysis

    Pointing loss and antenna gain calculations

- **Orbital Mechanics**

    Real satellite tracking using Two-Line Element (TLE) data

    Visibility analysis and pass prediction

    Distance and elevation calculations using Skyfield

- **Machine Learning Integration**

    Time-series prediction of signal fades

    Random Forest regression for fade forecasting

    Proactive system adaptation recommendations

    Performance benchmarking and model evaluation

- **Data Visualization**

    3D satellite constellation plots

    System performance dashboards

    Link quality analysis

    Ground station performance metrics

## The complete workflow is:

    mbse_model.py -> Defines the system blocks

    simulator.py -> Provides the simulation engine

    data_manager.py -> Supplies orbital data

    run_digital_twin.py -> Uses all above to run simulations

    visualization.py -> Creates graphs from simulation results

    ml_predictor.py -> Adds predictive capabilities