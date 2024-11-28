# elsciRL-Sailing

Application of the elsciRL system to a Sailing Simulator.

elsciRL-RL (private until further development)

Sailing Simulator:
 - https://github.com/topics/sailing-simulator
 - Simple sailing simulator from https://github.com/PPierzc/ai-learns-to-sail
   - https://github.com/PPierzc/ai-learns-to-sail/blob/master/tasks/channel.py

If setup correctly then the rest is handled by the elsciRL system
> Run main.py (change experiment call and parameters as needed)

# Instruction following

### Instruction: Descriptions

- dead center from left/right: dead center of the river
  NEW -> center from left/right: move to the middle of the river
- beach side: continue so that the boat is halfway between the center and the edge of the river but on the beach side with the wind on the port side of the boat
- turn back from beach: when the boat is halfway between the center and the edge of the river on the beach side, turn back towards the center so the wind is on the starboard side
- harbor side: continue so that the boat is halfway between the center and the edge of the river but on the harbor side with the wind on the starboard side of the boat
- turn back from harbor: when halfway between the center and the edge of the river on the harbor side, turn back towards the center so the wind is on the port side
- beach side -> re-specify to create cycle in meta-MDP
- Adds use of continuous instruction chaining -> prior sub-goals are now included in current env with reduced reward
- Sub-goals no longer terminate environment so agent can reach end goal if possible
- Total reward per episode can be increased by reaching sub-goals en-route to end goal 
- But need to ensure agent doesn't just go in circles for easy reward so either:
  - Reduce sub-goal reward (easy update in config) or,
  - Only first encounter gets the reward (requires further changes to env loop)






