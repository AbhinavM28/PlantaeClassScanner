# 🌿 PlantaeClassScanner

> An edge AI-powered handheld plant identification device with real-time classification running on Raspberry Pi

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)

![Project Status](https://img.shields.io/badge/status-in%20development-yellow)

---

## Project Overview

**PlantaeClassScanner** is a handheld, gun-styled plant identification device that uses computer vision and edge computing to classify plants in real-time. Designed as a portfolio project demonstrating skills in:

- Computer Vision & Deep Learning
- Embedded Systems & Edge Computing  
- Mechanical Design & CAD
- Full-Stack Product Development
- Hardware-Software Integration

### Key Features

- **Real-time Classification**: <500ms inference on Raspberry Pi 4B
- **Edge Computing**: All processing runs locally (no cloud dependency)
- **Interactive Hardware**: Realistic trigger mechanism with microswitch
- **Portable Design**: Battery magazine-style power supply
- **Expandable Dataset**: User-contributed plant scanning feature
- (Soon) **Health Detection**: Plant disease identification (Phase 2)
- (Soon) **Care Recommendations**: Actionable feedback for plant health

---

## Hardware Specifications

| Component | Specification |
|-----------|--------------|
| **Compute** | Raspberry Pi 4B (8GB RAM) |
| **Display** | 5" IPS Touchscreen (800x480, MIPI DSI) |
| **Trigger** | Microswitch (realistic gun trigger feel) |
| **Power** | Portable battery pack (magazine-style loading) |
| **Camera** | Raspberry Pi Camera Module v2/v3 |
| **Status LED** | GPIO-controlled feedback indicator |

---

## Model Architecture

- **Base Model**: MobileNetV3-Small (optimized for edge devices)
- **Input Size**: 224x224 RGB
- **Quantization**: INT8 (TensorFlow Lite)
- **Classes (v1.0)**: 10 common houseplants
- **Target Accuracy**: >85% on test set
- **Inference Time**: <500ms on RasPi 4B

---

## Dataset

### Initial Classes (v1.0)
1. Monstera deliciosa (Swiss Cheese Plant)
2. Epipremnum aureum (Pothos)
3. Sansevieria trifasciata (Snake Plant)
4. Chlorophytum comosum (Spider Plant)
5. Spathiphyllum wallisii (Peace Lily)
6. Ficus lyrata (Fiddle Leaf Fig)
7. Aloe vera
8. Crassula ovata (Jade Plant)
9. Hedera helix (English Ivy)
10. Ficus elastica (Rubber Plant)

### Data Sources
- **iNaturalist**: Research-grade observations
- **PlantNet-300K**: High-quality subset
- **User Contributions**: In-field scans (Phase 2)

**Target**: 100 images/class (1,000 total for v1.0)

---

## Quick Start

### Prerequisites
```bash
# Python 3.11
python3.11 --version

# Git LFS (for large files)
git lfs version
```

### Installation
```bash
# Clone repository
git clone https://github.com/AbhinavM28/PlantaeClassScanner.git
cd PlantaeClassScanner

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Download Initial Dataset
```bash
# Download 1,000 images (10 species × 100 images)
python scripts/download_dataset.py

# Split into train/val/test
python scripts/split_dataset.py

# Generate metadata
python scripts/generate_metadata.py
```

### Run Application (Development Mode)
```bash
# On WSL/Ubuntu (simulated hardware)
python src/main.py

# Press ENTER to simulate trigger
# Ctrl+C to exit
```

### Deploy to Raspberry Pi
```bash
# Transfer to RasPi via SCP
scp -r PlantaeClassScanner/ pi@raspberrypi.local:~/

# SSH into RasPi
ssh pi@raspberrypi.local

# Run on actual hardware
cd PlantaeClassScanner
source venv/bin/activate
python src/main.py
```

---

## 📁 Project Structure
```
PlantaeClassScanner/
├── data/                    # Dataset storage
│   ├── raw/                # Original downloaded images
│   ├── train/              # Training set (70%)
│   ├── val/                # Validation set (15%)
│   ├── test/               # Test set (15%)
│   └── dataset_metadata.json
├── models/                  # Model storage
│   ├── training/           # Training scripts
│   ├── checkpoints/        # Saved weights
│   └── deployed/           # Optimized TFLite models
├── src/                     # Source code
│   ├── data_collection/    # Dataset tools
│   ├── preprocessing/      # Image preprocessing
│   ├── training/           # Model training
│   ├── inference/          # Edge deployment
│   ├── hardware/           # GPIO/camera interface
│   ├── ui/                 # Display interface
│   └── utils/              # Config, logging, helpers
├── cad/                     # Mechanical design files
│   ├── enclosure/          # 3D printable housing
│   ├── components/         # Individual parts
│   └── assembly/           # Assembly instructions
├── docs/                    # Documentation
│   ├── hardware/           # Wiring diagrams, BOM
│   ├── assembly/           # Build instructions
│   └── performance/        # Benchmarks, metrics
├── tests/                   # Unit tests
├── scripts/                 # Utility scripts
├── config/                  # Configuration files
└── demo/                    # Demo videos/images
```

---

## Development Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [x] Repository setup with proper structure
- [x] Hardware abstraction layer (cross-platform dev)
- [x] Configuration management system
- [x] Professional logging system
- [x] Dataset collection pipeline
- [ ] Initial dataset (1,000 images) ← **IN PROGRESS**

### Phase 2: Model Development (Weeks 3-4)
- [ ] Data preprocessing pipeline
- [ ] Model training (MobileNetV3 transfer learning)
- [ ] Model optimization (TFLite quantization)
- [ ] Benchmark on RasPi 4B
- [ ] Achieve >85% test accuracy

### Phase 3: Hardware Integration (Week 5)
- [ ] GPIO trigger integration
- [ ] Camera module setup
- [ ] Display UI implementation
- [ ] Power management system
- [ ] Full hardware assembly

### Phase 4: CAD & Enclosure (Week 6)
- [ ] Enclosure design (Fusion 360)
- [ ] Trigger mechanism CAD
- [ ] Magazine-style battery holder
- [ ] 3D printing & assembly
- [ ] Fit testing & iteration

### Phase 5: Polish & Demo (Week 7)
- [ ] UI/UX refinements
- [ ] Performance optimization
- [ ] Demo video production
- [ ] Documentation completion
- [ ] Portfolio presentation materials

### Phase 6: Future Features
- [ ] User-contributed dataset expansion
- [ ] Plant health detection
- [ ] Disease identification
- [ ] Care recommendation system
- [ ] Mobile app companion

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| **Inference Time** | <500ms | TBD |
| **Model Accuracy** | >85% | TBD |
| **Model Size** | <50MB | TBD |
| **Power Consumption** | <2W idle, <5W active | TBD |
| **Battery Life** | >4 hours continuous | TBD |

---

## Contributing

This is a personal portfolio project, but feedback and suggestions are very much welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/suggestion`)
3. Commit changes (`git commit -m 'Add suggestion'`)
4. Push to branch (`git push origin feature/suggestion`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Abhinav M**

---

## About This Project

Built as a portfolio project to demonstrate expertise in:
- **Mechanical Engineering** with robotics focus
- **Product Design** from concept to prototype
- **Cross-functional Engineering** (hardware + software + design)

**Target Audience**: FAANG companies, top tech firms, robotics startups (If you like what you see, please reach out :D🙏)

**Contact**: 
- https://www.linkedin.com/in/abhimaddisetty/
- abhinav.maddisetty@outlook.com

---

## Acknowledgments

- **iNaturalist** - Community-driven biodiversity data
- **TensorFlow** - Edge AI framework
- **Raspberry Pi Foundation** - Accessible computing platform
- **PlantNet** - Plant identification research

---


**Status**: Active Development | Last Updated: October 2025

