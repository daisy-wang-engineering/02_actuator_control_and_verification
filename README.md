# 02_actuator_control_and_verification
## Single-Axis Actuator: Requirements-Driven Control and Verification

### Key Contributions
- Derived 16 verifiable requirements from 4 operational needs with a traceability matrix
- Verified each requirement by simulated test or analysis 
  - Bode margins, 200-run Monte Carlo, latency and fault-injection sweeps 
- Fault-injection testing exposed an undetected stuck-encoder hazard and 7 A blocked-rotor current against a 5 A rating
- Added plausibility monitor, fail-safe brake, limit switch and current limit, and re-verified
  - Worst-case excursion 64° vs 90° hard stop. 
- Documented a bandwidth-versus-latency requirements trade (CR-1) and three design iterations.
