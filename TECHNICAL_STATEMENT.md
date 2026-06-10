# Motivation & Approach

## Thought Process
While building the [RAG Research Extension](https://github.com/RobertTylman/RAG-Research-Extension), the need for a balanced research pipeline became apparent. I identified two critical gaps in the existing ecosystem:

1. **Research Mini is too slow for conversational UI**: While incredibly thorough, its standard execution time (30s–180s) introduces too much friction for continuous, interactive use.
2. **Search Advanced is too thin for complex schemas**: The standard fast search endpoint (~1–5s) lacks the multi-hop reasoning necessary to synthesize diverse sources and return deeply structured reports.

## The Thesis
Customers want deeply structured outputs and multi-hop reasoning capabilities, but they do not want to pay the high latency and cost penalties associated with a full Research Mini run.

## The Solution
A middle-ground product: **Tavily Research Mini Lite**. 

## Value Created
By compiling a bounded LangGraph agent loop that uses parallelized, single-hop Tavily `/search` calls and a post-hoc structured extraction, Research Mini Lite hits the "goldilocks zone." 

From both a technical and business perspective, it delivers synthesized, multi-hop, and structured research outputs while targeting a **~15 second** latency. This achieves the perfect midpoint between existing products, empowering conversational integrations to remain snappy while offering rigorous, structured intelligence.
