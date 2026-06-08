# Sourcing Agent — Simple Flow

## High-level architecture

```mermaid
flowchart LR
    User([User]) --> UI[Frontend]
    UI -->|POST prompt + URI| API[/auto-infer/]
    API --> Agent[Sourcing Agent Loop]

    Agent <-->|tool calls| Claude[(Claude)]
    Agent -->|read-only queries| Mongo[(MongoDB)]
    Agent -->|grounding DataFrame| Profile[Profile + SDV fit]
    Profile -->|SSE events| UI
    Agent -->|SSE events| UI
```

## What the agent actually does

```mermaid
flowchart TD
    Start([Prompt + Mongo URI]) --> Seed[Build seed message<br/>+ load SYSTEM_PROMPT<br/>+ TOOL_SCHEMAS]

    Seed --> Claude{{Claude<br/>claude-sonnet-4-6}}

    Claude -->|tool_use| Dispatch{Which tool?}

    Dispatch -->|list_collections| LC[list_collections]
    Dispatch -->|peek_schema| PS[peek_schema]
    Dispatch -->|count| CT[count]
    Dispatch -->|distinct_values| DV[distinct_values]
    Dispatch -->|finalize_grounding| FG[finalize_grounding]

    CT --> VF1[validate_filter<br/>operator whitelist]
    VF1 -->|reject| ErrResult[error result<br/>back to Claude]
    VF1 -->|pass| Mongo1[(MongoDB<br/>read-only)]
    LC --> Mongo1
    PS --> Mongo1
    DV --> Mongo1

    Mongo1 --> Summarize[Summarize result<br/>≤2000 chars JSON]
    Summarize --> SSE1[/SSE: step_start +<br/>step_complete/]
    SSE1 --> Append[Append tool_result<br/>to messages]
    ErrResult --> Append
    Append --> Claude

    FG --> Strategy[Capture rationale +<br/>query slices]
    Strategy --> SSE2[/SSE: rationale/]
    SSE2 --> Execute[execute_grounding]

    Execute --> VF2[validate_filter again<br/>+ clamp per-query 10k<br/>+ total budget 50k]
    VF2 --> Mongo2[(MongoDB<br/>read-only)]
    Mongo2 --> Union[Union slices into DataFrame<br/>tag with _segment label]
    Union --> Drop[Drop _segment col]
    Drop --> Profile[profile_dataframe_streaming]

    Profile --> SSE3[/SSE: schema_column ×N/]
    SSE3 --> SDV[Fit SDV<br/>GaussianCopulaSynthesizer]
    SDV --> SSE4[/SSE: final<br/>columns + stats + model_id +<br/>grounding_strategy + host/]
    SSE4 --> Done([Frontend assembles<br/>AgentTrace · SchemaCard ·<br/>GroundingStrategy])

    classDef llm fill:#3a3a5a,stroke:#7a7aaa,color:#fff
    classDef safety fill:#5a3a3a,stroke:#aa7a7a,color:#fff
    classDef sse fill:#3a5a3a,stroke:#7aaa7a,color:#fff
    classDef db fill:#5a4a3a,stroke:#aa8a6a,color:#fff
    class Claude llm
    class VF1,VF2 safety
    class SSE1,SSE2,SSE3,SSE4 sse
    class Mongo1,Mongo2 db
```

**Reading it:** the loop in the middle (`Claude → Dispatch → tool → Mongo → Append → Claude`) is the **discovery phase** — Claude keeps probing until it has enough to call `finalize_grounding`. Once it does, control drops out of the loop into the **execution phase** (right side), which re-validates everything and pulls the real grounding rows. The whole flow emits SSE events at every step so the frontend can build the agent trace and schema card live.

## The split: who controls what

```mermaid
flowchart LR
    subgraph Intent["Claude controls intent"]
        I1[Which collection?]
        I2[Which filter?]
        I3[Which slices to union?]
    end

    subgraph Mechanism["tools.py controls mechanism"]
        M1[find with _id projection]
        M2[Operator whitelist]
        M3[10k per-query cap]
        M4[50k total cap]
    end

    Intent --> Mechanism --> Mongo[(MongoDB)]
```

**Key idea:** Claude picks *what* to fetch, the tools enforce *how* — so the agent can act on a live cluster without write access, JS execution, or unbounded reads.
