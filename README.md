# GridflowAI: Smart Restoration Simulation with LLM Insight

---

## Overview

**GridflowAI** is an intelligent simulation and analysis tool designed to **quantify and optimize power restoration times** after grid outages.  
By combining **simulation data** with **LLM-based analysis**, GridflowAI enables energy planners to visualize restoration workflows, identify inefficiencies, and receive automated insights on how to improve recovery performance.

This project was developed for the **NextEra “Next100” Hackathon** to explore innovative AI-driven approaches to power grid resilience and restoration.

---

## Motivation

When a power outage occurs, restoration is a complex and multi-step process involving countless variables—crew deployment, repair priorities, resource allocation, and weather conditions.  
Currently, assessing “how long restoration should have taken” or “how it could have been improved” often relies on manual review.

**GridflowAI** changes that by:
- Simulating restoration sequences from start to finish,  
- Measuring restoration times under different conditions, and  
- Using an **LLM prompt engine** to automatically summarize and optimize the results.

---

## Key Features

- **Restoration Simulation:**  
  Generate time-based restoration progress based on custom parameters.

- **Quantitative Metrics:**  
  Compute restoration time, efficiency, and resource utilization from simulation data.

- **LLM-Driven Insight:**  
  Automatically interpret simulation results through a natural language prompt that identifies patterns and improvement strategies.

- **Replay & Optimization:**  
  “Replay” the restoration timeline to visualize performance and explore better restoration strategies.

---

## Example Use Case

1. Input simulation parameters (e.g., region size, outage scale, available crews).  
2. Run the model to simulate restoration progress.  
3. Feed the results into the LLM prompt to receive an automated summary like:

> “Restoration efficiency peaked at 73% after hour 4 due to optimal crew deployment. Consider pre-positioning additional teams near high-priority substations to reduce the initial delay.”

---

## Hackathon Context

This project was created as part of the **NextEra ‘Next100’ Hackathon**, focused on **innovation in energy resilience, grid modernization, and AI-driven analytics**.

Our goal was simple:  
> *If we could replay a restoration from start to finish, how could we improve it?*  

GridflowAI provides a quantitative and explainable answer.

