https://prod.liveshare.vsengsaas.visualstudio.com/join?FA3A5FC367C157BC87BF62730DD4EE71AF29




def prune_edges_crossing_buildings(G, buildings_all):
    """
    Remove roadmap edges that cross (or run through) any building polygon.
    Works with Shapely 1.x (returns geometries) and 2.x (returns indices).
    """
    if len(buildings_all) == 0:
        return G

    # Keep a list in the same positional order used to build the tree
    b_geoms = [g for g in buildings_all.geometry.values if g is not None and not g.is_empty]
    if not b_geoms:
        return G

    tree = STRtree(b_geoms)
    to_remove = []

    for u, v in list(G.edges()):
        p1 = Point(G.nodes[u]["x"], G.nodes[u]["y"])
        p2 = Point(G.nodes[v]["x"], G.nodes[v]["y"])
        seg = LineString([p1, p2])

        # Narrow candidates with a cheap bbox hit; Shapely 2.x returns indices here
        cand = tree.query(seg)  # optional: predicate="intersects" to prefilter more

        # Normalize candidates to actual polygon geometries
        if len(cand) == 0:
            continue
        if isinstance(cand[0], (int, np.integer)):
            polys = [b_geoms[i] for i in np.asarray(cand).tolist()]
        else:  # Shapely 1.x behavior: geometries directly
            polys = cand

        # Block if segment truly penetrates a polygon (touch-only is OK)
        hit = False
        for poly in polys:
            if seg.crosses(poly) or seg.within(poly) or (seg.intersects(poly) and not seg.touches(poly)):
                hit = True
                break
        if hit:
            to_remove.append((u, v))

    G.remove_edges_from(to_remove)
    return G

# ------------------ COMMS ------------------

def los_ok(pxy, qxy, occluder_tree, occluders_gs):
    seg = LineString([pxy, qxy])
    cand = occluder_tree.query(seg)  # optional: predicate="intersects"

    if len(cand) == 0:
        return True

    # Build the list used to create the tree, in positional order
    occ_geoms = [g for g in occluders_gs.values if g is not None and not g.is_empty]

    # Map indices -> geometries for Shapely 2.x; pass-through for 1.x
    try:
        first = cand[0]
    except Exception:
        return True

    if isinstance(first, (int, np.integer)):
        polys = [occ_geoms[i] for i in np.asarray(cand).tolist()]
    else:
        polys = cand

    for poly in polys:
        if seg.crosses(poly) or seg.within(poly) or seg.touches(poly):
            return False
    return True