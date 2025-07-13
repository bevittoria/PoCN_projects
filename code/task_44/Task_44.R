library(terra)
library(geodata)
library(dplyr)
library(eurostat)
library(sf)
library(countrycode)
library(readr)
library(purrr)
library(igraph)
library(ggplot2)

setwd("C:/Users/LENOVO PC/Dropbox/PC/Desktop/PoD/Complex networks/projects/44_facebook/gadm1_nuts3_counties-gadm1_nuts3_counties-fb-social-connectedness-index-october-2021")

read_facebook_data <- function(filepath) {
  read_tsv(filepath, show_col_types = FALSE) %>%
    filter(!grepl("USA", user_loc), !grepl("USA", fr_loc))
}

read_levels <- function(filepath) {
  read_csv(filepath, show_col_types = FALSE) %>%
    mutate(country = sub("^([A-Z]+).*", "\\1", key))
}

map_nuts3_countries <- function(levels_df) {
  nuts3_sf <- get_eurostat_geospatial(nuts_level = 3, year = 2016, output_class = "sf")
  nuts3_keys <- levels_df %>% filter(level == "nuts3") %>% pull(key)
  matched_countries <- nuts3_sf$CNTR_CODE[match(nuts3_keys, nuts3_sf$id)]
  iso3 <- countrycode(matched_countries, "eurostat", "iso3c")
  levels_df$country[levels_df$level == "nuts3"] <- iso3
  levels_df
}

prepare_main_df <- function(df, levels_df) {
  df %>%
    left_join(levels_df, by = c("user_loc" = "key")) %>%
    left_join(levels_df, by = c("fr_loc" = "key"), suffix = c("_user", "_fr")) %>%
    select(-level_fr) %>%
    rename(level = level_user) %>%
    filter(country_user == country_fr, level %in% c("gadm1", "gadm2", "nuts3")) %>%
    rename(country = country_fr) %>%
    select(-country_user)
}

sort_codes <- function(codes, level) {
  if (level == "gadm1") {
    codes[order(gsub("\\d", "", codes), as.integer(gsub("\\D", "", codes)))]
  } else if (level == "gadm2") {
    codes[order(as.integer(sub(".*_(\\d+)$", "\\1", codes)))]
  } else {
    sort(codes)
  }
}

build_country_network <- function(country, level, df_country, nuts3_shp) {
  node_codes <- unique(c(df_country$user_loc, df_country$fr_loc)) %>% sort_codes(level)
  
  if (level %in% c("gadm1", "gadm2")) {
    gadm_level <- ifelse(level == "gadm1", 1, 2)
    gadm_data <- tryCatch(gadm(country, level = gadm_level, version = "3.6", path = "./"), error = function(e) return(NULL))
    if (is.null(gadm_data) || nrow(gadm_data) == 0) return(NULL)
    
    node_ids <- if (level == "gadm1") {
      as.integer(gsub("\\D", "", node_codes))
    } else {
      as.integer(sub(".*_(\\d+)$", "\\1", node_codes))
    }
    
    names_vec <- if (level == "gadm1") gadm_data$NAME_1[node_ids] else gadm_data$NAME_2[node_ids]
    coords <- terra::crds(terra::centroids(gadm_data))[node_ids, , drop = FALSE]
    
    nodes <- tibble(
      nodeID = node_ids,
      name = names_vec,
      longitude = coords[, 1],
      latitude = coords[, 2],
      nodeCode = node_codes
    )
    
    get_nodeID <- function(code) nodes$nodeID[match(code, node_codes)]
    
    edges <- tibble(
      nodeID_from = map_int(df_country$user_loc, get_nodeID),
      nodeID_to   = map_int(df_country$fr_loc, get_nodeID),
      weight      = df_country$scaled_sci,
      country     = country
    )
    
  } else if (level == "nuts3") {
    nuts_regions <- nuts3_shp[match(node_codes, nuts3_shp$id), ]
    if (any(is.na(nuts_regions$geometry))) return(NULL)
    
    node_ids <- seq_along(node_codes)
    coords <- st_coordinates(st_centroid(nuts_regions$geometry))
    
    nodes <- tibble(
      nodeID = node_ids,
      name = nuts_regions$NAME_LATN,
      longitude = coords[, 1],
      latitude = coords[, 2],
      nodeCode = node_codes
    )
    
    get_nodeID <- function(code) nodes$nodeID[match(code, node_codes)]
    
    edges <- tibble(
      nodeID_from = map_int(df_country$user_loc, get_nodeID),
      nodeID_to   = map_int(df_country$fr_loc, get_nodeID),
      weight      = df_country$scaled_sci,
      country     = country
    )
  }
  
  list(nodes = nodes, edges = edges)
}

# === MAIN ===
df_raw <- read_facebook_data("gadm1_nuts3_counties.tsv")
levels_raw <- read_levels("gadm1_nuts3_counties_levels.csv")
levels_clean <- map_nuts3_countries(levels_raw)
df_clean <- prepare_main_df(df_raw, levels_clean)

country_levels <- df_clean %>% count(country, level) %>% select(country, level)
nuts3_shp <- get_eurostat_geospatial(nuts_level = 3, year = 2016, output_class = "sf")

network_list <- map2(country_levels$country, country_levels$level, function(cty, lvl) {
  df_cty <- df_clean %>% filter(country == cty)
  tryCatch(build_country_network(cty, lvl, df_cty, nuts3_shp), error = function(e) NULL)
})
names(network_list) <- country_levels$country

summary_df <- map_dfr(names(network_list), function(iso3) {
  net <- network_list[[iso3]]
  if (is.null(net)) return(NULL)
  
  nodes <- net$nodes %>% mutate(name = as.character(nodeID))
  edges <- net$edges %>% mutate(from = as.character(nodeID_from), to = as.character(nodeID_to))
  valid_ids <- nodes$name
  edges <- edges %>% filter(from %in% valid_ids, to %in% valid_ids)
  
  if (nrow(edges) == 0 || nrow(nodes) == 0) return(NULL)
  
  g <- igraph::graph_from_data_frame(edges, vertices = nodes, directed = FALSE)
  
  tibble(
    country = iso3,
    nodes = gorder(g),
    edges = gsize(g),
    density = edge_density(g),
    diameter = tryCatch(diameter(g), error = function(e) NA),
    avg_degree = mean(degree(g)),
    avg_clustering = mean(transitivity(g, type = "localundirected", isolates = "zero"), na.rm = TRUE),
    avg_path_length = tryCatch(mean_distance(g), error = function(e) NA),
    avg_weight = mean(edges$weight, na.rm = TRUE)
  )
}) %>% filter(avg_degree > 0)

# === PLOT ===
summary_df <- summary_df %>% mutate(edges_per_node = edges / nodes)

ggplot(summary_df, aes(x = nodes_tot, y = edges_tot)) +
  geom_point(size = 3, color = "tomato") +
  scale_y_log10() +scale_x_log10() +
  labs(title = "Nodes vs Edges", x = "Nodes", y = "Edges") +
  theme_minimal()

ggplot(summary_df, aes(x = nodes_tot, y = avg_path_length)) +
  geom_point(color = "forestgreen", size = 3) +
  scale_y_log10() +  scale_x_log10() +
  labs(title = "Total Nodes vs Average Path Length", x = "Total Nodes", y = "Average Path Length") +
  theme_minimal()

ggplot(summary_df, aes(y = avg_weight, x = avg_degree)) +
  geom_point(color = "darkred", size = 3) +
  scale_y_log10() + scale_x_log10() +
  labs(title = "Average Degree vs. Average SCI", x = "Average SCI", y = "Average Degree") +
  theme_minimal()

ggplot(summary_df, aes(y = avg_weight, x = nodes_tot)) +
  geom_point(color = "purple", size = 3) +
  scale_y_log10() +  scale_x_log10() +
  labs(title = "Average SCI vs. Total Nodes", x = "Average SCI", y = "Total Nodes") +
  theme_minimal()


