# 🚚 SmartLogix AI

## Intelligent Multi-Modal Logistics & Autonomous Delivery Platform

SmartLogix AI is an end-to-end AI-powered logistics platform designed to support intelligent order management, delivery tracking, fleet management, route optimization, drone delivery feasibility analysis, predictive maintenance, customer sentiment analysis, and AI-powered logistics decision-making.

The project combines Data Science, Machine Learning, Deep Learning, Computer Vision, Natural Language Processing, Generative AI, Retrieval-Augmented Generation (RAG), AI Agents, SQL, FastAPI, Streamlit, PostgreSQL, and AWS cloud deployment into a unified logistics platform.

---

# 📌 Project Objective

The objective of SmartLogix AI is to build an intelligent logistics platform that can:

- Analyze logistics and order data
- Clean and prepare multiple datasets
- Perform exploratory data analysis
- Predict logistics-related outcomes
- Classify suitable transport modes
- Predict delivery ETA
- Support predictive maintenance analysis
- Optimize delivery routes
- Support payload and vehicle selection
- Analyze fleet information
- Evaluate drone delivery feasibility
- Detect drone condition using Computer Vision
- Track delivery events
- Analyze customer reviews and sentiment
- Provide an AI-powered logistics assistant
- Retrieve relevant logistics information using RAG
- Provide an API for AI-agent interaction
- Provide an interactive Streamlit dashboard
- Support customer and employee workflows
- Deploy the application on AWS cloud infrastructure

---

# 🎯 Key Features

## 1. 📊 Data Understanding & Cleaning

SmartLogix AI works with multiple logistics datasets covering different areas of the delivery ecosystem.

The data preparation workflow includes:

- Data loading
- Data understanding
- Data type analysis
- Missing-value analysis
- Duplicate detection
- Data cleaning
- Data standardization
- Feature preparation
- Dataset integration

---

# 📂 Dataset Description

The project uses multiple logistics datasets covering different operational areas.

The datasets include:

- Order Information
- Product Details
- Customer Information
- Delivery Locations
- Weather Information
- Traffic Information
- Vehicle/Fleet Information
- Drone Sensor Data
- Drone Images
- Customer Reviews
- Delivery Logs

These datasets are used across data analysis, machine learning, route optimization, fleet analysis, delivery tracking, Computer Vision, NLP, and AI-powered logistics workflows.

---

# 2. 📈 Exploratory Data Analysis

Exploratory Data Analysis is performed across relevant logistics datasets to understand operational patterns and relationships.

EDA includes analysis of:

- Order patterns
- Delivery performance
- Customer behavior
- Product information
- Vehicle and fleet information
- Delivery events
- Transport modes
- Package weights
- Delivery distances
- Delivery performance
- Logistics relationships
- Operational patterns

Visualizations are used to identify trends, distributions, relationships, and potential data-quality issues.

---

# 3. 🤖 Machine Learning

SmartLogix AI includes machine learning models for logistics-related classification, prediction, and operational analysis.

The machine learning workflow includes:

- Feature preparation
- Data preprocessing
- Missing-value handling
- Categorical encoding
- Train/test splitting
- Model training
- Model evaluation
- Model comparison
- Model serialization
- Model integration with Streamlit

---

## 3.1 🚛 Transport Mode Classification

The system predicts a suitable logistics transport mode using the following input features:

- Quantity
- Package Weight
- Distance
- Order Value
- Delivery Priority
- Customer Segment
- Origin City
- Destination City

The transport modes include:

- Air Cargo
- Bike
- Drone
- Ship
- Truck
- Van

The final transport classification model uses a Random Forest classifier.

### Model Comparison

Two classification approaches were evaluated:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 62.23% |
| Random Forest | 88.35% |

The Random Forest model was selected for integration into the application based on the evaluation performed on the test dataset.

### Random Forest Classification Report

| Transport Mode | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Air Cargo | 0.92 | 0.83 | 0.87 |
| Bike | 0.88 | 0.90 | 0.89 |
| Drone | 0.66 | 0.63 | 0.64 |
| Ship | 0.40 | 0.08 | 0.14 |
| Truck | 0.89 | 0.97 | 0.93 |
| Van | 0.91 | 0.87 | 0.89 |

Overall accuracy:

**88.35%**

---

# 3.2 ⏱️ Delivery ETA Prediction

SmartLogix AI includes a machine learning model for delivery ETA prediction.

The model uses logistics-related information such as:

- Distance
- Vehicle information
- Package characteristics
- Weather
- Traffic
- Delivery information
- Operational information

The ETA model predicts the expected delivery duration in hours.

The trained ETA model is integrated into the Streamlit application for interactive prediction.

---

# 3.3 🛠️ Predictive Maintenance

SmartLogix AI includes a predictive maintenance module for vehicle/fleet analysis.

The maintenance model uses operational and vehicle-related features such as:

- Vehicle type
- Vehicle capacity
- Maximum range
- Average speed
- Ownership
- Vehicle capacity utilization
- Weight capacity condition
- Distance
- Delivery event count
- Failed attempt count
- RTO count
- Hub event count

A Random Forest classification model is used for predictive maintenance analysis.

The trained model is integrated into the Streamlit application.

---

# 4. 🗺️ Route Optimization

SmartLogix AI provides route optimization functionality to support efficient delivery planning.

The route optimization module considers logistics information such as:

- Origin
- Destination
- Distance
- Transport mode
- Vehicle information
- Delivery requirements
- Operational constraints

The Streamlit dashboard provides a map-based visualization of delivery routes.

Route information can be used to support logistics planning and delivery decision-making.

---

# 5. 📦 Payload Optimization

The system evaluates package weight and vehicle capacity to support appropriate vehicle and payload decisions.

The payload analysis considers:

- Package weight
- Vehicle capacity
- Vehicle type
- Delivery requirements
- Weight constraints

The system can identify whether the selected vehicle is appropriate for the package requirements.

---

# 6. 🚛 Fleet Management

The Fleet Management module provides information about vehicles used in the logistics operation.

It supports analysis of:

- Vehicle types
- Vehicle capacity
- Maximum range
- Average speed
- Fleet status
- Ownership
- Vehicle utilization
- Vehicle operational information

The fleet information is also used by other logistics modules such as predictive maintenance and payload analysis.

---

# 7. 🚁 Drone Management

The Drone Management module evaluates drone delivery feasibility using logistics and drone-related information.

The feasibility analysis considers:

- Package weight
- Drone capacity
- Delivery distance
- Maximum drone range
- Weight constraints
- Range constraints
- Delivery requirements

The system determines whether drone delivery can be considered for a particular logistics requirement.

---

# 8. 👁️ Computer Vision

SmartLogix AI includes a YOLO-based Computer Vision model for drone condition detection.

The model uses two classes:

- **Healthy Drone**
- **Damage Drone**

The Computer Vision workflow includes:

- Drone image dataset preparation
- Image preprocessing
- Annotation preparation
- Class mapping
- YOLO model training
- Model validation
- Object detection
- Confidence-based prediction
- Model integration with Streamlit

The Computer Vision model is integrated into the Drone Damage Detection section of the application.

---

# 9. 📝 NLP & Sentiment Analysis

Customer reviews are processed using Natural Language Processing techniques.

The system performs:

- Review text processing
- Text feature extraction
- Sentiment classification
- Positive/negative sentiment analysis
- Customer review insights

The sentiment analysis functionality helps understand customer feedback and review patterns.

---

# 10. 🧠 RAG + LLM

SmartLogix AI includes a Retrieval-Augmented Generation pipeline.

The RAG workflow is:

```text
User Question
      ↓
Intent Detection
      ↓
Relevant Logistics Data Retrieval
      ↓
Context Construction
      ↓
LLM
      ↓
Final Response

--

## 11. AI Agent

The SmartLogix AI Agent connects user questions with appropriate tools.

Supported query types include:

- Order queries
- Capacity queries
- Range queries
- Feasibility queries

Example:

"What is the status of ORD-005379?"

The agent identifies the query intent and selects the appropriate tool.

---

## 12. FastAPI Backend

FastAPI provides the backend API layer for SmartLogix AI.

The backend exposes API endpoints for:

- Health checking
- AI-agent interaction
- Logistics queries

Swagger API documentation is available through:

    http://127.0.0.1:8000/docs

---

## 13. Streamlit Dashboard

The Streamlit application provides an interactive interface for the
SmartLogix AI platform.

### Available Pages

- Dashboard
- Orders
- Delivery Tracking
- Fleet Management
- Route Optimization
- Drone Management
- Analytics
- AI Assistant
- Products
- Reviews & Insights
- Notifications

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      User           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Streamlit UI       │
                    │  SmartLogix AI      │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       ┌───────────┐     ┌────────────┐    ┌─────────────┐
       │ Analytics │     │ AI Agent   │    │ Operations  │
       └───────────┘     └─────┬──────┘    └─────────────┘
                               │
                               ▼
                      ┌────────────────┐
                      │    FastAPI     │
                      └───────┬────────┘
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
           ┌──────────┐ ┌──────────┐ ┌──────────┐
           │ RAG      │ │ Ollama / │ │ PostgreSQL│
           │ Pipeline │ │ DeepSeek │ │ Database  │
           └──────────┘ └──────────┘ └──────────┘
                 │
                 ▼
          Logistics Datasets
