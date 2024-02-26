# Crime Detection System Readme

## Current System
The current system operates as follows:
- The enforcer manually collects data from witnesses and suspects.
- Natural intelligence is used to identify potential criminals.

### Challenges of Current System
- Slow and error-prone due to manual data collection.
- Lack of technology integration limits analysis and suspect identification accuracy.

## Proposed Idea

### Overview
We propose a crime detection system focusing on cyber security cases. This system involves three actors: the law enforcer, suspect, and witness.

### System Architecture Diagram

![System Architecture Diagram](system_architecture_diagram.png)

### System Features
- **Landing Page**: Includes authentication and role selection.
- **Enforcer Dashboard**: Allows adding/viewing case details and statements.
- **Suspect Dashboard**: Enables testimony and case report viewing.
- **Witness Dashboard**: Facilitates testimony.

### User Authentication
- Users receive an authorization code via email for access.
- Role selection determines dashboard access.

### Case Management
- Enforcer adds case details: ID, description, suspects, witnesses, and status.
- Suspects and witnesses are added based on enforcer input.
- Enforcer writes statement including incident details.
- Enforcer can view case reports.

### Sample Enforcer Dashboard Interface

![Enforcer Dashboard Interface](enforcer_dashboard.png)

### Testimony
- Suspects and witnesses provide testimony through system-generated questions.
- Testimonies are stored and analyzed for consistency.

### Data Analysis
- Sentiment analysis calculates emotion, confidence, and consistency scores.
- Machine learning model predicts potential criminals based on suspect data and testimony.

### Use Case Model
- User accesses landing page and authenticates.
- Role determines dashboard access.
- Enforcer manages cases and statements.
- Suspects and witnesses testify.
- Data analysis and prediction inform case reports.

## Conclusion
The proposed crime detection system integrates technology to streamline case management and improve suspect identification accuracy.
