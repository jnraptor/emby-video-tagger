# Emby Video Tagger Architecture Diagrams

## Current Architecture (Monolithic)

```mermaid
graph TD
    A[emby_video_tagger.py<br/>924 lines] --> B[TaskStatus Enum]
    A --> C[EmbyVideoTagger Class]
    A --> D[IntelligentFrameExtractor Class]
    A --> E[BaseVisionProcessor ABC]
    A --> F[LMStudioVisionProcessor]
    A --> G[OllamaVisionProcessor]
    A --> H[VisionProcessorFactory]
    A --> I[VideoTaggingAutomation]
    A --> J[main function]
    
    I --> C
    I --> D
    I --> H
    H --> F
    H --> G
    F --> E
    G --> E
    
    style A fill:#ff6666
```

## Proposed Architecture (Modular)

```mermaid
graph TB
    subgraph "Entry Points"
        CLI[cli.py]
        MAIN[__main__.py]
    end
    
    subgraph "Configuration Layer"
        CONFIG[config/settings.py]
        VALIDATORS[config/validators.py]
    end
    
    subgraph "Core Domain"
        MODELS[core/models.py]
        INTERFACES[core/interfaces.py]
        EXCEPTIONS[core/exceptions.py]
    end
    
    subgraph "Service Layer"
        EMBY[services/emby.py]
        EXTRACTOR[services/frame_extractor.py]
        ORCHESTRATOR[services/orchestrator.py]
        
        subgraph "Vision Processing"
            VBASE[vision/base.py]
            VLMSTUDIO[vision/lmstudio.py]
            VOLLAMA[vision/ollama.py]
            VFACTORY[vision/factory.py]
        end
    end
    
    subgraph "Data Layer"
        DB[storage/database.py]
        DBMODELS[storage/models.py]
    end
    
    subgraph "Infrastructure"
        LOGGING[utils/logging.py]
        PATHMAPPER[utils/path_mapper.py]
        SCHEDULER[scheduler/jobs.py]
    end
    
    CLI --> ORCHESTRATOR
    MAIN --> CLI
    
    ORCHESTRATOR --> EMBY
    ORCHESTRATOR --> EXTRACTOR
    ORCHESTRATOR --> VFACTORY
    ORCHESTRATOR --> DB
    
    VFACTORY --> VLMSTUDIO
    VFACTORY --> VOLLAMA
    VLMSTUDIO --> VBASE
    VOLLAMA --> VBASE
    
    EMBY --> CONFIG
    ORCHESTRATOR --> CONFIG
    
    DB --> DBMODELS
    
    style ORCHESTRATOR fill:#90EE90
    style CONFIG fill:#87CEEB
    style VFACTORY fill:#DDA0DD
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Orchestrator
    participant EmbyService
    participant FrameExtractor
    participant VisionProcessor
    participant Database
    
    User->>CLI: Run command
    CLI->>Orchestrator: Process videos
    
    Orchestrator->>EmbyService: Get recent videos
    EmbyService-->>Orchestrator: Video list
    
    loop For each video
        Orchestrator->>Database: Check if processed
        Database-->>Orchestrator: Processing status
        
        alt Not processed
            Orchestrator->>FrameExtractor: Extract frames
            FrameExtractor-->>Orchestrator: Frame paths
            
            par Concurrent frame analysis
                Orchestrator->>VisionProcessor: Analyze frame 1
                Orchestrator->>VisionProcessor: Analyze frame 2
                Orchestrator->>VisionProcessor: Analyze frame N
            end
            
            VisionProcessor-->>Orchestrator: Tags
            
            Orchestrator->>EmbyService: Update video tags
            EmbyService-->>Orchestrator: Success
            
            Orchestrator->>Database: Update status
        end
    end
    
    Orchestrator-->>CLI: Processing complete
    CLI-->>User: Results
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "External Systems"
        EMBY_SERVER[Emby Server]
        AI_SERVER[AI Server<br/>LMStudio/Ollama]
        FILE_SYSTEM[File System]
    end
    
    subgraph "Application Core"
        CONFIG_MGR[Configuration<br/>Manager]
        ORCH[Orchestrator]
        TASK_REPO[Task<br/>Repository]
    end
    
    subgraph "Service Adapters"
        EMBY_ADAPTER[Emby<br/>Adapter]
        VISION_ADAPTER[Vision<br/>Adapter]
        FRAME_SERVICE[Frame<br/>Service]
    end
    
    EMBY_SERVER <--> EMBY_ADAPTER
    AI_SERVER <--> VISION_ADAPTER
    FILE_SYSTEM <--> FRAME_SERVICE
    
    ORCH --> EMBY_ADAPTER
    ORCH --> VISION_ADAPTER
    ORCH --> FRAME_SERVICE
    ORCH --> TASK_REPO
    
    CONFIG_MGR --> ORCH
    CONFIG_MGR --> EMBY_ADAPTER
    CONFIG_MGR --> VISION_ADAPTER
```

## Performance Optimization Strategy

```mermaid
graph TD
    subgraph "Current Sequential Processing"
        A1[Video 1] --> B1[Extract Frames] --> C1[Analyze Frame 1]
        C1 --> C2[Analyze Frame 2]
        C2 --> C3[Analyze Frame N]
        C3 --> D1[Update Tags]
        D1 --> A2[Video 2]
        A2 --> B2[Extract Frames]
        B2 --> C4[Analyze Frames]
        C4 --> D2[Update Tags]
    end
    
    subgraph "Proposed Concurrent Processing"
        V1[Video 1] --> E1[Extract Frames]
        V2[Video 2] --> E2[Extract Frames]
        V3[Video N] --> E3[Extract Frames]
        
        E1 --> F1[Frame 1.1]
        E1 --> F2[Frame 1.2]
        E1 --> F3[Frame 1.N]
        
        E2 --> F4[Frame 2.1]
        E2 --> F5[Frame 2.2]
        E2 --> F6[Frame 2.N]
        
        F1 --> G1[Analyze]
        F2 --> G2[Analyze]
        F3 --> G3[Analyze]
        F4 --> G4[Analyze]
        F5 --> G5[Analyze]
        F6 --> G6[Analyze]
        
        G1 --> H1[Merge Tags]
        G2 --> H1
        G3 --> H1
        
        G4 --> H2[Merge Tags]
        G5 --> H2
        G6 --> H2
        
        H1 --> I1[Update Video 1]
        H2 --> I2[Update Video 2]
    end
    
    style A1 fill:#ff6666
    style V1 fill:#90EE90
    style V2 fill:#90EE90
    style V3 fill:#90EE90
```

## Error Handling Flow

```mermaid
flowchart TD
    START[Start Processing] --> TRY[Try Operation]
    
    TRY --> SUCCESS{Success?}
    SUCCESS -->|Yes| CONTINUE[Continue Processing]
    SUCCESS -->|No| ERROR_TYPE{Error Type?}
    
    ERROR_TYPE -->|Emby API Error| RETRY_API{Retry Count < 3?}
    ERROR_TYPE -->|Frame Extraction Error| FALLBACK[Use Fallback Extraction]
    ERROR_TYPE -->|Vision Processing Error| SKIP_FRAME[Skip Frame]
    ERROR_TYPE -->|Critical Error| LOG_CRITICAL[Log Critical Error]
    
    RETRY_API -->|Yes| BACKOFF[Exponential Backoff]
    RETRY_API -->|No| MARK_FAILED[Mark Video Failed]
    
    BACKOFF --> TRY
    FALLBACK --> CONTINUE
    SKIP_FRAME --> CONTINUE
    LOG_CRITICAL --> MARK_FAILED
    
    MARK_FAILED --> UPDATE_DB[Update Database]
    UPDATE_DB --> NEXT_VIDEO[Process Next Video]
    
    CONTINUE --> COMPLETE{All Done?}
    COMPLETE -->|No| TRY
    COMPLETE -->|Yes| END[End Processing]