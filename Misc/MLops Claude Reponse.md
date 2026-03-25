# MLOps Final Project Deep-Dive Analysis

**Bottom Line Up Front:** Of the 10 project ideas evaluated, **Vision-Based Anomaly Detection, IoT Predictive Maintenance, and Edge AI Optimization** emerge as the strongest candidates for an April 15th deadline—offering excellent public datasets, manageable GCP costs ($200-800/month), and clear MLOps demonstration potential. **F1 Strategy Optimization** and **Fashion Recommendation** are also feasible with moderate scope reduction. Four projects face significant blockers: **Pharmaceutical ML**, **LimbScan**, and **EMS Speech-to-Text** suffer from severe data availability and regulatory constraints, while **Interior Redesign** requires computational resources exceeding typical academic budgets.

---

## Comparative feasibility matrix

| Project | Data Availability | Regulatory Complexity | GCP Cost (Monthly) | Timeline Risk | **Overall Score** |
|---------|-------------------|----------------------|-------------------|---------------|-------------------|
| **Vision Anomaly Detection** | ✅ Excellent | Low | $200-800 | Low | **9/10** |
| **IoT Predictive Maintenance** | ✅ Excellent | Low | $260-760 | Low | **9/10** |
| **Edge AI Optimization** | ✅ Good | Low | $125-385 | Low | **8/10** |
| **F1 Strategy** | ✅ Good | Low | $126-452 | Medium | **8/10** |
| **Fashion Recommendation** | ✅ Good | Low | $200-500 | Medium | **7/10** |
| **Camera Calibration** | ✅ Good | Low | $63-210 | Low | **6/10*** |
| **Interior Redesign** | ⚠️ Moderate | Low | $400-3,000 | High | **5/10** |
| **Pharmaceutical ML** | ❌ Limited | Very High | $200-700 | High | **4/10** |
| **LimbScan Prosthetics** | ❌ Very Limited | Very High | $50-700 | High | **4/10** |
| **EMS Speech-to-Text** | ❌ No Public Audio | High | $1,585-3,330 | Very High | **3/10** |

*Camera calibration scores lower because ML adds limited value over traditional methods*

---

## 1. F1 Strategy Optimization

### Technical architecture delivers strong prediction potential

This project leverages **FastF1**, an unofficial but robust Python library providing lap timing, telemetry (speed, throttle, brake, DRS, gear), tire data, and weather information from 2018 onwards. The **Jolpica API** (successor to the deprecated Ergast API) supplies historical race results since 1950. Real teams use billions of simulations per weekend, but academic research shows **CNN-LSTM models achieve 96% accuracy** for pit stop prediction, while **reinforcement learning (DQN)** outperformed Monte Carlo baselines at the 2023 Bahrain GP.

**Key technical pros**: Rich pandas DataFrame integration, 300+ sensors per car generating 1.1M+ data points/second during races, excellent caching for efficient development. **Key technical cons**: No raw AWS/F1 partnership data access, fuel load data unavailable, ±10m telemetry synchronization errors, ~3-second real-time latency.

**Solutions implemented in research**: Distance-normalized data instead of time-normalized to handle sync errors; fuel estimation at ~0.03s/kg from lap time progression; Monte Carlo simulation for stochastic safety car events; SMOTE for class imbalance in pit decisions (achieving F1-score of 0.81).

### Competitor landscape spans commercial to open-source

AWS F1 Insights powers 20+ broadcast graphics but remains inaccessible to external developers. **Open-source alternatives** include FastF1 (3k+ GitHub stars), OpenF1 API, and TracingInsights archives. Academic approaches from Mercedes collaboration (RSRL using DQN) and Heilmeier et al.'s Virtual Strategy Engineer demonstrate production viability.

### Public datasets enable immediate development

- **Jolpica API**: http://api.jolpi.ca/ergast/f1/ (1950-present results)
- **FastF1 Library**: https://docs.fastf1.dev/ (2018+ telemetry)
- **OpenF1 API**: https://openf1.org/ (2023+ real-time/historical)
- **Kaggle F1 Championship**: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

### GCP costs remain modest at $126-452/month

| Service | Estimate |
|---------|----------|
| Vertex AI Training (T4) | $40-80 |
| Cloud Storage (50GB) | $1-2 |
| Cloud Run inference | $20-50 |
| BigQuery analytics | $5-25 |
| Model serving endpoint | $50-100 |

### PAIR Framework assessment favors augmentation

**Prediction suitability**: High—clear targets (pit windows, position changes) with measurable outcomes. **Personalization**: Good fit for driver/team-specific models using historical performance data. **Automation vs. augmentation**: Must be augmentation—FIA rules require human decision-making, and millions of dollars at stake demand accountability. Implement SHAP values for explainability.

### April 15th feasibility: ✅ Achievable

MVP scope: Historical data pipeline → pit window prediction model → Cloud Run API → basic Streamlit dashboard. Timeline allows 10 weeks with buffer.

---

## 2. Fashion Recommendation System (India Focus)

### Computer vision achieves 89-96% accuracy on fashion classification

Pre-trained models (ResNet50, YOLOv8) and benchmark datasets enable rapid prototyping. **Shein's AI-driven recommendations account for 26% of revenue** with 85% trend prediction accuracy. Alta generates 270,000+ outfits daily using 12+ specialized models. The India market presents massive opportunity with Meesho's **200M+ MAUs and 9% e-commerce market share**.

**Key challenges**: Cultural/regional diversity (traditional Indian attire underrepresented in Western datasets), rapid trend evolution, cold-start problem, style subjectivity.

**Solutions**: Multi-modal approach combining visual + text (CLIP embeddings); style profile quiz (Stitch Fix collects 90 data points); collaborative filtering; continuous learning pipeline like Shein's LATR model.

### Strong competitor ecosystem provides benchmarks

| Competitor | Technology | Key Insight |
|------------|------------|-------------|
| **Alta** | 12+ AI models, RLHF | $11M seed; 270K outfits/day |
| **Stitch Fix** | 4.5B textual data points | AI drives 75% of selections |
| **Meesho** | BharatMLStack (open-sourced) | 66.9T feature retrievals annually |
| **Shein** | Hyper-personalization | Trend-to-product in 48 hours |

### Public datasets available but lack Indian fashion

- **DeepFashion**: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html (800K+ images)
- **DeepFashion2**: https://github.com/switchablenorms/DeepFashion2 (491K images)
- **Polyvore Outfits**: https://github.com/xthan/polyvore-dataset (21,889 outfits)
- **Fashion-MNIST**: https://github.com/zalandoresearch/fashion-mnist
- **Fashionpedia**: https://fashionpedia.github.io/home/

⚠️ **No large-scale Indian fashion dataset exists publicly**—recommend fine-tuning on curated images or partnering with retailers.

### GCP costs scale with usage at $200-500 for MVP

| Scenario | Monthly Cost |
|----------|--------------|
| MVP/Academic | $100-300 |
| Small Scale (10K users) | $500-1,500 |
| Medium Scale (100K users) | $2,000-5,000 |

### PAIR assessment supports hybrid automation

**Personalization** is the core value proposition—individual wardrobe analysis with continuous feedback loops. **Augmentation model recommended** (like Stitch Fix's 1,600 stylists + AI). Must address bias monitoring for body types and cultural preferences.

### April 15th feasibility: ✅ Achievable with focused MVP

Focus on: wardrobe photo upload → automatic categorization → basic outfit compatibility → weather-based recommendations. Skip virtual try-on and price comparison for Phase 2.

---

## 3. Edge AI Model Optimization

### Compression techniques achieve 4x-100x model reduction

**Quantization** (FP32→INT8) delivers 4x size reduction with 97-99% accuracy retention. **Pruning** achieves up to 57% reduction at 46% speed increase. **Knowledge distillation** enables 5-50x compression. Combined pipelines reach **87% shrinkage with 65% speed improvement**.

| Platform | Memory | Compute | Power | Key Constraints |
|----------|--------|---------|-------|-----------------|
| MCUs (Cortex-M4/M7) | 128KB-1MB | 80-216 MHz | <100mW | INT8 only |
| Google Coral Edge TPU | 8MB SRAM | 4 TOPS | 2W | TFLite required |
| NVIDIA Jetson Nano | 4GB | 472 GFLOPS | 5-10W | Broader support |

### Mature competitor ecosystem provides tools

- **Edge Impulse**: EON Compiler achieves 25-55% less RAM than TFLite
- **TensorFlow Lite/TF Micro**: Broad hardware support, Google ecosystem
- **Apache TVM**: Auto-tuning, µTVM for MCUs
- **ONNX Runtime**: Cross-framework, <1MB mobile version

### MLPerf Tiny benchmarks enable standardized evaluation

- **Image Classification**: CIFAR-10, 85% Top-1 target — https://www.cs.toronto.edu/~kriz/cifar.html
- **Visual Wake Words**: COCO-derived, 80% Top-1 target — https://github.com/mlcommons/tiny
- **Keyword Spotting**: Speech Commands, 90% Top-1 target — https://www.tensorflow.org/datasets/catalog/speech_commands
- **Anomaly Detection**: ADMOS, 0.85 AUC target — https://github.com/mlcommons/tiny

### GCP costs remain low at $125-385/month

Training on T4 GPUs (~$0.73/hr), storage for datasets (~$3/month), Cloud Build CI/CD ($1.50-5/month), Artifact Registry for model versions (~$6/month).

### PAIR framework supports automated optimization with human oversight

**Prediction**: Performance prediction for different hardware saves engineering time. **Personalization**: Hardware-specific optimization profiles. **Automation vs. augmentation**: 70% automated for routine compression, 30% human-guided for critical tradeoffs.

### April 15th feasibility: ✅ Highly achievable

Focus on quantization pipeline (PTQ + QAT) for 2 models (MobileNet, DS-CNN), deployment to Coral USB, basic CI/CD, MLPerf Tiny benchmark evaluation.

---

## 4. Vision-Based Anomaly Detection for Quality Control

### Unsupervised methods solve the rare-defect data problem

**PatchCore achieves 99.1% image AUROC** on MVTec AD benchmark using memory banks and nearest-neighbor approaches. **PaDiM** (Gaussian patch distributions) requires no training. Only defect-free images needed, enabling detection of novel, unseen defect types.

**Key advantages**: Works with sparse defect data; **FastFlow achieves 30-50ms inference** for real-time requirements; well-established benchmarks enable reproducible research.

### Commercial competition validates market opportunity

| Company | Position | Key Feature |
|---------|----------|-------------|
| **Cognex** | Industry leader | Edge learning with 5-10 images |
| **Landing AI** | Andrew Ng's company | Data-centric MLOps, $57M funding |
| **Instrumental** | Electronics focus | High-mix NPI inspection |

⚠️ **Amazon Lookout for Vision discontinuing October 31, 2025**—migration to SageMaker required.

### Best-in-class public datasets available

| Dataset | Size | URL |
|---------|------|-----|
| **MVTec AD** | 5,354 images, 15 categories | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| **MVTec AD 2** (2025) | 8,000+ images | https://www.mvtec.com/company/research/datasets/mvtec-ad-2 |
| **VisA (Amazon)** | 10,821 images | https://registry.opendata.aws/visa/ |
| **DAGM 2007** | 16,100 images | https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection |
| **Severstal Steel** | 18,074 images | https://www.kaggle.com/c/severstal-steel-defect-detection |

**Anomalib** (OpenVINO Toolkit) provides 20+ SOTA algorithms with MVTec support: https://github.com/open-edge-platform/anomalib

### GCP costs scale reasonably at $200-800/month for development

| Component | Estimate |
|-----------|----------|
| Vertex AI Training (T4/V100) | $73-250 |
| Cloud Storage (image data) | $5-10 |
| Cloud Run/GKE inference | $50-200 |
| Monitoring | $10-30 |

### PAIR assessment supports full automation for high-volume production

**Prediction**: 95-99% AUROC achievable. **Personalization**: Product-specific models using MVTec's category structure. **Automation**: Full automation appropriate for high-volume, low-variance production; human-in-loop for high-mix scenarios.

### April 15th feasibility: ✅ Highly achievable (11 weeks)

Use Anomalib + PatchCore, MVTec AD dataset, Cloud Run deployment, Pub/Sub streaming pipeline. Strong MLOps demonstration potential.

---

## 5. IoT Predictive Maintenance

### CNN-LSTM hybrid models achieve 96.1% accuracy on benchmarks

Time-series approaches vary in effectiveness: **CNN-LSTM hybrid** leads (96.1% accuracy, 95.2% F1-score), followed by LSTM (98.57% on specific datasets), with Transformers showing 10% improvement in some POCs but requiring more data.

**Key challenges**: Extreme class imbalance (228,416 healthy vs. 8 failures in some datasets), sensor data preprocessing, multi-sensor fusion complexity.

**Solutions**: SMOTE + Wasserstein GAN (EO-WGAN achieves 95.2% accuracy); LSTM encoder-decoder for missing value imputation (R²=0.90); autoencoder preprocessing for irrelevant sensor filtering.

### Enterprise competitors dominate but open-source viable

- **Uptake**: Fleet maintenance, Samsara integration, 25% lower maintenance costs
- **Azure IoT + Predictive Maintenance**: Full reference architecture available
- **IBM Maximo Predict**: 30-40% reduction in unplanned downtime
- **Note**: Google Cloud IoT Core deprecated August 2023—use Pub/Sub + Vertex AI alternative

### NASA and academic datasets provide excellent training data

| Dataset | Description | URL |
|---------|-------------|-----|
| **NASA C-MAPSS** | Turbofan engine degradation, 21 sensors | https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip |
| **CWRU Bearing** | Vibration data, 12-48K samples/sec | https://engineering.case.edu/bearingdatacenter/download-data-file |
| **FEMTO Bearing** | 17 run-to-failure tests | https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip |
| **Kaggle Azure PM** | Microsoft's predictive maintenance | https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance |

### GCP costs optimize via Pub/Sub BigQuery subscription

**Key insight**: Direct Pub/Sub → BigQuery subscription is **10x cheaper than Dataflow** for streaming.

| Component | Monthly Estimate |
|-----------|------------------|
| Vertex AI inference | $50-200 |
| Cloud Storage (500GB) | $10 |
| BigQuery (10TB queries) | $50-100 |
| Pub/Sub → BigQuery | $50-150 |
| **Total** | **$260-760** |

### PAIR framework recommends augmentation for maintenance planning

**Prediction**: Clear targets (failure probability, RUL in cycles/days). **Personalization**: Machine-specific models learning from each asset's history. **Automation**: Automated alerts appropriate; human-in-loop for maintenance scheduling decisions.

### April 15th feasibility: ✅ Highly achievable (10 weeks)

Use NASA C-MAPSS FD001, CNN-LSTM hybrid, Pub/Sub BigQuery subscription, Streamlit/Grafana dashboard.

---

## 6. Pharmaceutical ML – Critical barriers identified

### Data availability is the primary blocker

**Historical pharma data is "fragmented, inconsistent, and proprietary."** One study had to manually extract 1,982 formulations from 232 academic papers. No comprehensive public dissolution profile datasets exist.

### Regulatory complexity exceeds academic project scope

- **21 CFR Part 11**: Full system validation (IQ/OQ/PQ protocols) required
- **FDA**: No specific guidance for AI in pharmaceutical manufacturing (as of 2025)
- **EU Annex 22**: Being developed, signaling increased scrutiny
- Most applications remain "at proof-of-concept or early translational stage with limited clinical validation"

### Available public databases focus on chemistry, not formulation

| Resource | URL | Limitation |
|----------|-----|------------|
| **ChEMBL** | https://www.ebi.ac.uk/chembl/ | Bioactivity, not formulation |
| **PubChem** | https://pubchem.ncbi.nlm.nih.gov/docs/downloads | Structure data |
| **DrugBank** | https://go.drugbank.com/releases/latest | Drug properties |
| **FDA Orange Book** | https://catalog.data.gov/dataset/approved-drug-products-with-therapuetic-equivalence-evaluations-orange-book | Approved products |
| **FDT Dataset** | BMC Research Notes paper | 1,982 formulations (manually extracted) |

### GCP costs manageable but compliance adds overhead

$50-200/month for MVP; $300-700 for HIPAA-compliant prototype; $1,000-3,000+ for GMP-ready infrastructure.

### PAIR assessment: Augmentation only (regulatory mandate)

Black-box ML conflicts with regulatory explainability requirements. Must use interpretable models (Random Forest + SHAP).

### April 15th feasibility: ⚠️ Challenging

**Recommendation**: Pivot to tablet disintegration time prediction using the BMC FDT dataset. Frame as "research demonstration" with clear disclaimers about clinical applicability.

---

## 7. Camera Calibration – ML adds limited value

### Traditional methods already achieve sub-pixel accuracy

Zhang's method achieves **RMS error ~0.03-0.5 pixels** with proper setup. OpenCV provides mature, well-tested implementation. Kalibr (ETH Zurich, 5.2k GitHub stars) handles multi-camera + IMU calibration with rolling shutter support.

### ML is useful only for specific scenarios

- **Online/continuous calibration** in autonomous vehicles
- **Targetless calibration** from natural scenes
- **LiDAR-Camera extrinsic calibration** (where ML genuinely excels)

### Public datasets exist but problem is largely solved

- **KITTI**: https://www.cvlibs.net/datasets/kitti/ (autonomous driving benchmark)
- **TUM VI**: Visual-inertial odometry
- **EuRoC MAV**: MAV dataset with ground truth

### GCP costs minimal at $63-210/month

### PAIR assessment: ML suitability is limited

**Camera calibration is largely a solved problem for offline use.** ML adds genuine value only for automation (eliminating calibration workflow) or multi-modal sensor fusion.

### April 15th feasibility: ✅ Achievable but lower impact

**Recommendation**: If pursuing, focus on **LiDAR-Camera extrinsic calibration** using KITTI—this is where ML genuinely outperforms traditional methods.

---

## 8. Interior Redesign – Computational barriers significant

### Technology has matured but remains expensive

**Gaussian Splatting (August 2023)** now outperforms NeRF: faster training, real-time rendering, editable scenes. **InteriorGS dataset** (2025) provides 1,000 indoor scenes in 3DGS format.

**Critical costs**: A100 GPU at ~$2.50-3.00/hr; one full NeRF scene = $10-50+; video storage adds up quickly (1 min 4K ≈ 500MB-1GB).

### Modsy's failure provides cautionary lessons

Raised $72.7M but **shut down June 2022** citing "capital constraints." High computational costs created unsustainable economics. B2C furniture margins don't support complex 3D tech.

### Competitors use simpler 2D approaches

| Company | Approach | Status |
|---------|----------|--------|
| **RoomGPT** | 2D AI redesign, single photo | Active, 2M+ users |
| **Havenly** | AI + human designers | Market leader |
| **Planner 5D** | 2D/3D floor planning | Active |

### Public datasets available for 3D indoor scenes

- **ScanNet++**: https://arxiv.org/abs/2308.11417 (460 scenes, 280K DSLR images)
- **Matterport3D**: https://github.com/niessner/Matterport (90 buildings)
- **Hypersim**: https://github.com/apple/ml-hypersim (461 synthetic scenes)
- **InteriorGS**: https://github.com/manycore-research/InteriorGS (1,000 3DGS scenes)

### GCP costs range from $400-3,000/month

Full production with heavy training + inference reaches $1,500-3,000/month—exceeding typical academic budgets.

### April 15th feasibility: ❌ Very challenging

**Strong recommendation**: Pivot to **2D image-based approach** using pre-trained diffusion models (Stable Diffusion + ControlNet). This is achievable in 2.5 months and demonstrates MLOps concepts without prohibitive compute costs.

---

## 9. LimbScan Prosthetics – Data and regulatory barriers insurmountable

### Real clinical need but data access impossible for students

Only 58% of sockets in low-income countries provide adequate fit. AI research shows **1.24mm median surface-to-surface distance error** achievable. But: no IRB approval, no HIPAA-compliant data access, no public residual limb datasets.

### Major prosthetics companies lead research

- **Ottobock**: €1.6B revenue, 2,600+ patents, IPO Oct 2025
- **Össur**: Acquired Standard Cyborg's CAD software
- **e-NABLE**: 40,000+ volunteers, 10,000-15,000 device recipients (3D printed, open-source)

### Body shape datasets exist but lack residual limb data

- **SMPL/SMPL-X**: https://smpl.is.tue.mpg.de (free for research)
- **CAESAR**: ~$5,000-25,000 from SAE
- **NIH Visible Human Project**: https://www.nlm.nih.gov/research/visible/getting_data.html

**No public dataset exists for residual limb 3D scans or socket designs.**

### GCP costs manageable but HIPAA adds complexity

$50-200/month for synthetic-data-only approach; $300-700 with HIPAA compliance.

### PAIR assessment: Must be prosthetist augmentation

Socket failures cause falls—amputees fall at 200x the rate of the general population. AI must support, not replace, clinical expertise.

### April 15th feasibility: ⚠️ Challenging

**Recommendation**: Reframe as "Synthetic Residual Limb Shape Modeling: A Technical Feasibility Study." Use SMPL truncation to generate synthetic data. Focus on demonstrating MLOps pipeline, not clinical tool.

---

## 10. EMS Speech-to-Text – No public audio data exists

### NEMSIS provides excellent structured target schema

Version 3.5.1 uses HL7 standards, SNOMED procedure codes, RxNorm medications, ICD-10 diagnoses. Documentation available at https://nemsis.org/technical-resources/version-3/version-3-data-dictionaries/

### Medical ASR challenges compound in EMS environments

**Word Error Rates**: AWS Medical achieved ~39% median WER on healthcare conversations; Whisper showed 72-75% WER. EMS adds siren interference, radio static, multiple speakers, vehicle noise.

### Competition validates market gap

**ESO and ImageTrend dominate EMS documentation but neither offers speech-to-text integration.** Ambient clinical documentation startups (Abridge at $2.75B valuation, Nabla, Suki) focus on clinical settings, not emergency services.

- **MedSpaCy**: https://github.com/medspacy/medspacy (clinical NLP)
- **scispaCy**: https://allenai.github.io/scispacy/ (biomedical text)

### Critical finding: No publicly available EMS audio datasets

HIPAA restrictions, operational sensitivity, and lack of consent protocols in emergency settings prevent public data release.

### GCP costs high at $1,585-3,330/month production

Speech-to-Text Medical API at ~$1,280-2,560/month for 10,000 calls; plus Vertex AI NER pipeline.

### April 15th feasibility: ❌ Major blockers

**Recommendation**: Use synthetic audio generation (TTS + noise augmentation) as primary data source. Frame as "proof of concept" with clear documentation of limitations. Focus on MLOps pipeline demonstration, not production-ready clinical tool.

---

## Strategic recommendations by team profile

### For teams prioritizing low risk and clear deliverables
**Choose Vision-Based Anomaly Detection or IoT Predictive Maintenance.** Both have excellent public datasets (MVTec AD, NASA C-MAPSS), established benchmarks, manageable costs ($200-800/month), and clear 11-week timelines. These demonstrate production MLOps practices without regulatory complexity.

### For teams seeking differentiation with acceptable risk
**Choose F1 Strategy Optimization or Fashion Recommendation.** Both have good data availability but require thoughtful scoping. F1 benefits from strong open-source community (FastF1); Fashion offers India market differentiation but lacks Indian fashion datasets.

### For teams with strong ML backgrounds seeking challenge
**Choose Edge AI Optimization.** Requires understanding of quantization, pruning, and knowledge distillation but has clear benchmarks (MLPerf Tiny) and hardware targets (Coral, Jetson). Demonstrates advanced MLOps with model versioning for multiple deployment targets.

### Projects requiring significant scope reduction
- **Camera Calibration**: Pivot to LiDAR-Camera extrinsic calibration
- **Interior Redesign**: Pivot to 2D image-based approach
- **Pharmaceutical ML**: Focus on tablet disintegration prediction only
- **LimbScan**: Use synthetic data, frame as feasibility study
- **EMS Speech-to-Text**: Use synthetic audio, document limitations

---

## GCP cost summary across all projects

| Project | Development | MVP Production | Notes |
|---------|-------------|----------------|-------|
| Vision Anomaly Detection | $100-300 | $500-800 | Best value |
| IoT Predictive Maintenance | $150-300 | $260-760 | Pub/Sub optimization |
| Edge AI Optimization | $100-200 | $125-385 | Lowest cost |
| F1 Strategy | $50-150 | $126-452 | Good value |
| Fashion Recommendation | $100-200 | $200-500 | Scales with users |
| Camera Calibration | $50-100 | $63-210 | Minimal requirements |
| Interior Redesign | $200-500 | $400-3,000 | GPU-intensive |
| Pharmaceutical ML | $50-150 | $200-700 | Plus compliance overhead |
| LimbScan | $50-100 | $50-700 | Synthetic data only |
| EMS Speech-to-Text | $100-200 | $1,585-3,330 | Highest cost |

All estimates assume GCP's $300 free credits for new accounts (90 days) can offset initial development costs.

---

## Conclusion

The April 15th deadline creates a hard constraint favoring projects with existing public datasets and established benchmarks. **Vision-Based Anomaly Detection** and **IoT Predictive Maintenance** emerge as the strongest candidates, combining excellent data availability (MVTec AD, NASA C-MAPSS), reasonable costs ($200-800/month), and clear MLOps demonstration potential. **Edge AI Optimization** offers the lowest costs and clearest benchmarks (MLPerf Tiny). **F1 Strategy** and **Fashion Recommendation** are viable with focused scoping.

Projects requiring medical data (Pharmaceutical ML, LimbScan, EMS Speech-to-Text) face insurmountable barriers for student teams—no public datasets, complex regulatory requirements, and high compliance costs. **Interior Redesign** exceeds typical academic compute budgets. These should be pursued only with significant scope reduction and explicit framing as technical feasibility studies rather than production-ready systems.