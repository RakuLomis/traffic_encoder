# Traffic Encoder

**Choices**: 
- Use Pyshark to analyze the xml extracted from Wireshark? 
- Use self-defined fields extractor to analyze? 

### Updates

**2025.07.18**: 
- Optimize the performance of handling pcap files. 
  - Finally, the program can continuously run till the end. However, the speed is fluctuant in different kinds of pcaps. 

**2025.07.20**: 


### Pipeline

#### Merge and Field Block Truncation

1. label_extract.py: It gets the label by extracting the name of directory and merges the csvs with label if needed. 
2. pcap_processing.ipynb: It merges the different kinds of csvs into a completed csv. 