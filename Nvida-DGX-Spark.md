# NVIDIA DGX Spark Configuration Guide

## Overview
This guide describes how to configure an **NVIDIA DGX Spark** system. It includes setup steps for system initialization

---

## Prerequisites
Before starting, ensure the following:
- NVIDIA DGX Spark system is powered and connected.
- Internet connectivity configured (for updates and downloads). (Make sure the WIFI being used is open and only requires password, with no terms to accept once connected)

---

## Steps: Initial Setup (First Boot) using a Laptop 
1. Power on the DGX Spark and follow the first-boot wizard.
2. The system creates a wifi hotspot that you will need to connect to on your laptop, the ssid and passwotf are on the cover of the quick start guide. E.g spark-6s23 
3. Once connected, open the system setup page **http://device-name.local** e.g http://spark-6s23.local
4. Accept the terms once logged in.
5. Select your **language**, **keyboard layout**, and **timezone**. America/New York
6. Create an **administrator account** (avoid reserved names like `sysadmin` or `admin`).
7. After that you need to connect the system to a wifi network so that the system updates.
8. After update is done you can log on to the system

## System update
1. Make sure that system is updated on machine first start - "sudo apt update && sudo apt upgrade"
2. Install personal/project requirements  

## SSH
1. IP address can be found using ifconfig or you can use the name of the device as the identifier 
e.g ssh username@spark-6s23.local  or ssh username@10.10.10.1
2. Use your password you create in the step 6 above to connect.



