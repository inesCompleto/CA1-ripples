
# CA1 Ripples

## Overview

This repository contains computational modelling and data analysis tools related to sharp-wave ripples (SWRs) in the hippocampal CA1 region.

Sharp-wave ripples are transient network events occurring in the hippocampus, characterised by a large-amplitude sharp wave accompanied by high-frequency oscillations (ripples). They are particularly important for hippocampal-dependent memory consolidation, spatial navigation and the reactivation of previously encoded neuronal activity patterns.

The aim of this project is to investigate CA1 ripple activity through complementary computational modelling and data analysis approaches, providing tools to explore the mechanisms underlying these events and their associated neuronal dynamics.

## Project status

**Work in progress**

This project is currently under active development. In particular, the computational model of hippocampal CA1 sharp-wave ripples is still being developed and has not yet reached its final implementation.

The repository is intended to document the ongoing research and provide access to the current codebase. Scripts, model components and analysis pipelines may be subject to changes as the project progresses.

Please note that the current model should be considered a work in progress and that its implementation and results may not yet reflect the final objectives of the project.

## Repository structure

The repository is organised into two main directories:

```text
CA1-ripples/
│
├── Data-analysis/
│   └── ...                  # Data analysis scripts
│
├── Model/
│   └── ...                  # Computational model (under development)
│
└── README.md
```

### Data-analysis

This directory contains the scripts and tools used to analyse data related to CA1 ripple activity.

The analysis component provides a framework for investigating electrophysiological activity and characterising neuronal dynamics associated with sharp-wave ripple events.

### Model

This directory contains the computational modelling component of the project.

The model aims to investigate the neuronal and network mechanisms underlying sharp-wave ripple generation and dynamics in hippocampal CA1.

**This component is currently under development.** Its implementation, parameters and functionality are subject to change as the project progresses.

## Scientific background

Sharp-wave ripples (SWRs) are one of the most prominent oscillatory patterns observed in the hippocampus.

They consist of two main components:

- **Sharp waves:** Large, slower deflections in the local field potential (LFP), generally associated with excitatory input from CA3 to CA1.
- **Ripples:** Fast oscillations superimposed on the sharp-wave component, reflecting coordinated neuronal activity within the CA1 network.

SWRs are thought to play an important role in memory consolidation through the reactivation and replay of neuronal activity sequences.

Understanding the mechanisms underlying their generation and organisation is therefore essential for investigating hippocampal network dynamics.

## Project objectives

The project aims to study CA1 sharp-wave ripples through complementary experimental data analysis and computational modelling approaches.

The main objectives are:

- Investigate the mechanisms underlying CA1 sharp-wave ripple generation.
- Explore neuronal and network dynamics associated with ripple events.
- Analyse electrophysiological activity related to sharp-wave ripples.
- Develop a computational framework to investigate the mechanisms underlying CA1 ripple activity.

## Installation

Clone the repository:

```bash
git clone https://github.com/inesCompleto/CA1-ripples.git

cd CA1-ripples
```

The required software dependencies depend on the scripts and computational framework used in each directory.

Please refer to the individual scripts and notebooks for additional dependency information.

## Usage

The repository contains two complementary components.

### 1. Data analysis

Navigate to the data analysis directory:

```bash
cd Data-analysis
```

The scripts in this directory can be used to perform analyses associated with CA1 ripple activity.

Please refer to the individual scripts for information about the required input data, analysis parameters and expected outputs.

### 2. Computational modelling

Navigate to the model directory:

```bash
cd Model
```

This directory contains the computational model currently being developed to investigate CA1 network activity and ripple dynamics.

As the model is still under development, its implementation and usage instructions may change. Please refer to the corresponding scripts and documentation for the current implementation.

## Data and results

Input data and generated results should be organised according to the requirements of the individual analysis and modelling scripts.

Please refer to the corresponding code for information about input file formats, data structures and output organisation.

## Notes and considerations

- The repository contains separate components for computational modelling and data analysis.
- The data analysis and computational modelling components may have different software dependencies.
- The computational model is currently under development and should not be considered a final implementation.
- Model parameters, scripts and analysis settings may change as the project progresses.
- Results should be interpreted in the context of the biological mechanisms underlying hippocampal sharp-wave ripples.
