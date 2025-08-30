#!/usr/bin/env python3
"""
AirSim Drone Status Checker
Standalone script to check if drones are armed and ready
"""

import airsim
import time

def check_drone_status():
    """Check the status of all drones in AirSim"""
    
    print("🔍 Checking AirSim Drone Status...")
    print("=" * 50)
    
    try:
        # Connect to AirSim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✅ Connected to AirSim successfully!")
        
        # Get list of all vehicles
        vehicles = client.listVehicles()
        print(f"📋 Found {len(vehicles)} vehicles: {vehicles}")
        print()
        
        if not vehicles:
            print("❌ No vehicles found in AirSim!")
            return
        
        # Check each drone's status
        for drone_name in vehicles:
            print(f"🚁 Checking {drone_name}:")
            print("-" * 30)
            
            try:
                # Get drone state
                state = client.getMultirotorState(drone_name)
                
                # Check basic status
                print(f"  Connected: {state.connected}")
                print(f"  Ready: {state.ready}")
                print(f"  Armed: {state.ready}")  # Ready = Armed in AirSim
                
                # Check API control
                try:
                    api_control = client.isApiControlEnabled(drone_name)
                    print(f"  API Control: {api_control}")
                except Exception as e:
                    print(f"  API Control: Error checking - {e}")
                
                # Check if drone can move
                if state.ready:
                    print(f"  Status: ✅ READY TO FLY")
                    
                    # Test if it can actually move
                    print(f"  Testing movement capability...")
                    try:
                        # Get current position
                        pos_before = client.simGetVehiclePose(drone_name).position
                        print(f"    Position before: [{pos_before.x_val:.3f}, {pos_before.y_val:.3f}, {pos_before.z_val:.3f}]")
                        
                        # Try a small movement command
                        print(f"    Sending test movement command...")
                        cmd = client.moveByVelocityBodyFrameAsync(0.1, 0.0, 0.0, duration=0.5, vehicle_name=drone_name)
                        cmd.join()
                        
                        # Check if it moved
                        pos_after = client.simGetVehiclePose(drone_name).position
                        movement = ((pos_after.x_val - pos_before.x_val)**2 + 
                                  (pos_after.y_val - pos_before.y_val)**2 + 
                                  (pos_after.z_val - pos_before.z_val)**2)**0.5
                        
                        print(f"    Position after: [{pos_after.x_val:.3f}, {pos_after.y_val:.3f}, {pos_after.z_val:.3f}]")
                        print(f"    Movement: {movement:.6f}")
                        
                        if movement > 0.001:
                            print(f"    ✅ Movement test PASSED")
                        else:
                            print(f"    ❌ Movement test FAILED - No movement detected")
                            
                    except Exception as move_error:
                        print(f"    ❌ Movement test ERROR: {move_error}")
                        
                else:
                    print(f"  Status: ❌ NOT READY")
                    
                    # Try to arm it
                    print(f"  Attempting to arm {drone_name}...")
                    try:
                        # Enable API control first
                        client.enableApiControl(True, vehicle_name=drone_name)
                        time.sleep(0.5)
                        
                        # Try to arm
                        client.armDisarm(True, vehicle_name=drone_name)
                        time.sleep(1.0)
                        
                        # Check if arming worked
                        new_state = client.getMultirotorState(drone_name)
                        if new_state.ready:
                            print(f"    ✅ Successfully armed {drone_name}!")
                        else:
                            print(f"    ❌ Failed to arm {drone_name}")
                            
                    except Exception as arm_error:
                        print(f"    ❌ Arming error: {arm_error}")
                
                print()
                
            except Exception as e:
                print(f"  ❌ Error checking {drone_name}: {e}")
                print()
        
        print("=" * 50)
        print("🔍 Status check complete!")
        
    except Exception as e:
        print(f"❌ Failed to connect to AirSim: {e}")
        print("Make sure AirSim is running and accessible!")

if __name__ == "__main__":
    check_drone_status()
