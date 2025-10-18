<!-- 828f9535-69fa-4d44-8b70-1ed15daa5ff7 d961b0e0-3b24-4ab5-b9b0-8f542d6bb895 -->
# Transform to Surf Trip Planner

## Overview

Convert the general travel planner into a specialized surf trip planner targeting intermediate/advanced surfers. Keep the 4-agent architecture but refocus all agents, tools, and data on surf travel.

## Key Files to Modify

### 1. Backend Core (`backend/main.py`)

**Agent Transformations:**

- **Research Agent** → **Surf Spot Research Agent**: Gather wave conditions, break characteristics, best seasons
- **Budget Agent** → **Surf Budget Agent**: Calculate costs including board rentals, surf lessons, wax, repairs
- **Local Agent** → **Local Surf Culture Agent**: Provide surf etiquette, local scene insights, surf shops
- **Itinerary Agent** → **Surf Trip Itinerary Agent**: Create day-by-day plans with dawn patrol sessions, tide timing

**Tool Transformations:**

- `essential_info()` → `surf_spot_info()`: Break type, wave characteristics, best swell direction
- `weather_brief()` → `surf_forecast_brief()`: Swell size/direction, wind conditions, tides
- `visa_brief()` → Keep but add surf gear customs info
- `budget_basics()` → `surf_trip_budget()`: Include board rentals, lessons, wax, wetsuit costs
- `attraction_prices()` → `surf_services_pricing()`: Surf lessons, guided sessions, boat trips
- `local_flavor()` → `local_surf_scene()`: Surf culture, local shapers, competitions
- `local_customs()` → `surf_etiquette()`: Lineup rules, localism awareness, respect protocols
- `hidden_gems()` → `secret_spots()`: Lesser-known breaks for advanced surfers

**State Schema Updates:**

- Update `TripState` to include surf-specific fields
- Update `TripRequest` model to include skill_level, board_preference fields

### 2. Surf Spots Database (`backend/data/local_guides.json` → `surf_spots.json`)

Create new database with 25 famous surf destinations:

- **Format**: destination, break_name, skill_levels, wave_type, best_season, swell_direction, crowds, description, source
- **Destinations**: Include Pipeline (Hawaii), J-Bay (South Africa), Teahupo'o (Tahiti), Uluwatu (Bali), Trestles (California), etc.
- **Coverage**: Mix of reef breaks, point breaks, beach breaks across skill levels

### 3. Frontend UI (`frontend/index.html`)

**Rebrand:**

- Title: "Plan Your Perfect Surf Trip"
- Update hero image to surf-focused (waves, surfers)
- Update icons to surf-relevant (waves, surfboard, compass)

**Form Modifications:**

- Destination field: Keep but update placeholder to "e.g., North Shore, Hawaii"
- Duration field: Keep
- Budget field: Update options (Budget Surfer, Mid-Range, Premium Surf Resort)
- Replace generic "Interests" with "Surf Preferences": reef breaks, point breaks, barrel hunting, mellow cruising
- Optional: Add skill level selector (Intermediate / Advanced / Expert)

**Feature Cards:** Update 4 cards:

- Research → Surf Spot Intel
- Itineraries → Session Planning  
- Budget → Surf Trip Costs
- Local → Local Surf Scene

### 4. Documentation Updates

**README.md:**

- Update title to "AI Surf Trip Planner"
- Replace all travel references with surf travel
- Update architecture diagram descriptions
- Modify example use cases to surf-focused scenarios
- Update API endpoint examples with surf destinations
- Change "trip planner" to "surf trip planner" throughout

**IMPLEMENTATION_SPEC.md:**

- Update agent descriptions to surf focus
- Modify tool call examples
- Update performance metrics context

**API_INTEGRATION_SPEC.md:**

- Add section on future surf forecast APIs (Surfline, Magicseaweed, Stormglass)
- Update example integrations to surf-relevant services
- Keep as future enhancement documentation

**RAG.md** (if exists):

- Update to describe surf spot vector search
- Explain curated surf spot recommendations

### 5. Other Files

- `start.sh`: Update any displayed text if present
- `render.yaml`: Update service name/description if needed

## Implementation Approach

### Phase 1: Core Backend (Most Critical)

1. Update all agent prompts in `main.py` for surf focus
2. Transform all 8-10 tools to surf-specific versions
3. Update LLM fallback prompts for surf contexts
4. Modify state types and request/response models

### Phase 2: Surf Spots Database

1. Create `backend/data/surf_spots.json`
2. Research and add 25 world-class surf destinations
3. Include diverse break types, seasons, skill levels
4. Update RAG retriever to use surf_spots.json

### Phase 3: Frontend Transformation

1. Update all text, titles, descriptions
2. Change hero images and branding
3. Modify form fields for surf preferences
4. Update feature cards and icons

### Phase 4: Documentation

1. Update README.md with surf branding
2. Revise IMPLEMENTATION_SPEC.md
3. Update API_INTEGRATION_SPEC.md with surf APIs
4. Update any other markdown files

## Key Terminology Changes

- "Trip" → "Surf Trip"
- "Destination" → "Surf Destination" / "Break"
- "Attractions" → "Surf Spots" / "Breaks"
- "Local experiences" → "Local surf scene"
- "Itinerary" → "Surf Trip Itinerary" / "Session Plan"
- "Research" → "Surf Spot Intel"

## Testing After Changes

1. Run backend: `./start.sh` or `uvicorn main:app --reload`
2. Test with surf destination: "Pipeline, North Shore, Hawaii"
3. Verify all 4 agents execute with surf-focused responses
4. Check frontend displays surf terminology
5. Test RAG retrieval finds relevant surf spots

### To-dos

- [x] Transform all tools in backend/main.py to surf-specific versions (surf_spot_info, surf_forecast_brief, surf_trip_budget, etc.)
- [x] Update all 4 agent prompts and logic for surf focus (Research→Surf Spot Research, Budget→Surf Budget, Local→Local Surf Culture, Itinerary→Surf Trip)
- [x] Update state types, request/response models with surf-specific fields and terminology
- [x] Create surf_spots.json with 25 famous surf destinations (Pipeline, J-Bay, Uluwatu, etc.) and update retriever
- [x] Transform frontend/index.html: rebrand to surf trip planner, update hero, modify form fields for surf preferences
- [x] Update README.md with surf trip planner branding, examples, and terminology
- [x] Update IMPLEMENTATION_SPEC.md and API_INTEGRATION_SPEC.md for surf focus

