# elsciRL Application: Sailing Simulation

<div align="center">
  <b>Application of the elsciRL system to a Sailing Simulator.</b>
  <br>
  Visit our <a href="https://elsci.org">website</a> to get started, explore our <a href="https://github.com/pdfosborne/elsciRL-Wiki">open-source Wiki</a> to learn more or join our <a href="https://discord.gg/GgaqcrYCxt">Discord server</a> to connect with the community.
</div>

<div align="center">  
  <br>

  <a href="https://github.com/pdfosborne/elsciRL">![elsciRL GitHub](https://img.shields.io/github/watchers/pdfosborne/elsciRL?style=for-the-badge&logo=github&label=elsciRL&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL)</a>
  <a href="https://github.com/pdfosborne/elsciRL-Wiki">![Wiki GitHub](https://img.shields.io/github/watchers/pdfosborne/elsciRL-Wiki?style=for-the-badge&logo=github&label=elsciRL-Wiki&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL-Wiki)</a>
  <a href="https://discord.gg/GgaqcrYCxt">![Discord](https://img.shields.io/discord/1310579689315893248?style=for-the-badge&logo=discord&label=Discord&link=https%3A%2F%2Fdiscord.com%2Fchannels%2F1184202186469683200%2F1184202186998173878)</a>

</div>

## Sailing Simulator

Sourced from:

 - https://github.com/topics/sailing-simulator
 - Simple sailing simulator from https://github.com/PPierzc/ai-learns-to-sail
   - https://github.com/PPierzc/ai-learns-to-sail/blob/master/tasks/channel.py

![sailing_overview](https://raw.githubusercontent.com/pdfosborne/elsciRL-App-Sailing/refs/heads/main/prerender/sailing_setup.png)

# Quick Experiment

If setup correctly then the rest is handled by the elsciRL system
> Run main.py (change experiment call and parameters as needed)


# Instruction following

Instructions do not terminate environment so agent can reach end goal if possible. Total reward per episode can be increased by reaching sub-goals en-route to end goal. But need to ensure agent doesn't just go in circles for easy reward so only first encounter gets the reward.

### Examples

**Beach Side**

- go to the beach side
- continue so that the boat is halfway between the center and the edge of the river but on the beach side with the wind on the port side of the boat
- when the boat is halfway between the center and the edge of the river on the beach side, turn back towards the center so the wind is on the starboard side

**Harbor Side**
- go to the harbor side
- continue so that the boat is halfway between the center and the edge of the river but on the harbor side with the wind on the starboard side of the boat
- when halfway between the center and the edge of the river on the harbor side, turn back towards the center so the wind is on the port side



---
last-commit-id:aa7dacb (Used to trigger GUI import)

---



