---
title: Technical Details
layout: default
nav_order: 7
---

# Technical Details
Details describing the implementation of the viewer will be added here. Example flowchart using Mermaid.


```mermaid
flowchart TB
    remote_viewer<-->local_viewer
    subgraph server
    renderer --> local_viewer
    end
    subgraph client
    remote_viewer
    end

```

## Rendering pipeline

## Scene Components
Scene, Nodes, and Renderables

## How we handle transparency





