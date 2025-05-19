import pandas as pd
import matplotlib.pyplot as plt

# Optimized consolidation with product-level detail, KPI, sustainability metrics, and visualizations
def main():
    # Load data
    orders_file = 'D:/project_mostrans/data/historical_sales.xlsx'
    forecast_file = 'D:/project_mostrans/outputs/forecast_7d.xlsx'

    vehicles = pd.read_excel(orders_file, sheet_name='vehicles')
    distances = pd.read_excel(orders_file, sheet_name='distances')
    weights = pd.read_excel(orders_file, sheet_name='product_weights')
    forecast = pd.read_excel(forecast_file, sheet_name='forecast')
    forecast['date'] = pd.to_datetime(forecast['date'])

    # Map each outlet to nearest warehouse
    dist_min = distances.loc[distances.groupby('outlet')['distance_km'].idxmin()]
    outlet_wh = dict(zip(dist_min['outlet'], dist_min['warehouse']))
    forecast['warehouse'] = forecast['outlet'].map(outlet_wh)

    # Use first forecast date
    date0 = forecast['date'].min()
    shipments = forecast[forecast['date'] == date0].copy()

    # Compute weight per product
    shipments = shipments.merge(weights, on='product')
    shipments['boxes'] = shipments['forecast_qty_box']
    shipments['weight_kg'] = shipments['boxes'] * shipments['weight_per_box_kg']
    shipments = shipments.merge(distances, on=['warehouse', 'outlet'])

    # Pending shipments at product-outlet level
    pending = shipments[['warehouse','outlet','product','boxes','weight_per_box_kg','weight_kg','distance_km']].to_dict('records')

    # Consolidation logic: multi-trip, splitting
    consolidation = []
    trip = 1
    while pending:
        # reset capacities per warehouse
        veh_caps = {}
        for r in pending:
            wh = r['warehouse']
            if wh not in veh_caps:
                dfv = vehicles[vehicles['warehouse']==wh].copy()
                veh_caps[wh] = dfv.assign(remaining=dfv['max_weight_kg']).to_dict('records')
        # assign each warehouse
        next_pending = []
        for wh, caps in veh_caps.items():
            wh_pending = [r for r in pending if r['warehouse']==wh]
            wh_pending.sort(key=lambda x: x['weight_kg'], reverse=True)
            for rec in wh_pending:
                boxes_left = rec['boxes']
                wpb = rec['weight_per_box_kg']
                dist = rec['distance_km']
                for veh in sorted(caps, key=lambda x: x['remaining'], reverse=True):
                    if boxes_left<=0:
                        break
                    max_boxes = int(veh['remaining']//wpb)
                    if max_boxes<=0:
                        continue
                    assigned = min(boxes_left, max_boxes)
                    wgt = assigned * wpb
                    consolidation.append({
                        'trip': trip,
                        'warehouse': wh,
                        'vehicle_name': veh['vehicle_name'],
                        'outlet': rec['outlet'],
                        'product': rec['product'],
                        'boxes': assigned,
                        'weight_kg': wgt,
                        'distance_km': dist
                    })
                    veh['remaining'] -= wgt
                    boxes_left -= assigned
                if boxes_left>0:
                    next_pending.append({**rec, 'boxes': boxes_left, 'weight_kg': boxes_left*wpb})
        pending = next_pending
        trip += 1

    df_opt = pd.DataFrame(consolidation)

    # Compute utilization per trip & vehicle
    util = df_opt.groupby(['trip','vehicle_name'])['weight_kg'].sum().reset_index()
    util = util.merge(vehicles[['vehicle_name','max_weight_kg']], on='vehicle_name')
    util['util_pct'] = util['weight_kg']/util['max_weight_kg']*100

    # Compute weight, trips, and distance
    total_trips = df_opt['trip'].nunique()
    total_weight = df_opt['weight_kg'].sum()
    dist_map = df_opt.groupby(['trip','vehicle_name'])['distance_km'].max().reset_index()
    dist_map['roundtrip_km'] = dist_map['distance_km']*2
    total_distance = dist_map['roundtrip_km'].sum()

    # KPI metrics
    avg_util = util['util_pct'].mean()
    avg_delivs = df_opt.groupby('trip')['outlet'].nunique().mean()
    avg_weight_trip = df_opt.groupby('trip')['weight_kg'].sum().mean()
    density = total_weight/total_distance if total_distance>0 else float('nan')

    # Sustainability: CO2 emissions
    # Emission factor per vehicle type
    ef = {'Truk CDE':0.8, 'Truk CDD':0.8, 'Pick Up':0.4}
    # Attach vehicle type
    dist_map = dist_map.merge(vehicles[['vehicle_name','vehicle_type']], on='vehicle_name')
    dist_map['ef'] = dist_map['vehicle_type'].map(ef)
    dist_map['co2_kg'] = dist_map['roundtrip_km'] * dist_map['ef']
    total_co2 = dist_map['co2_kg'].sum()
    co2_per_kg = total_co2 / total_weight if total_weight>0 else float('nan')

    # Performance summary
    perf = {
        'Total Trips': total_trips,
        'Total Weight (kg)': total_weight,
        'Total Distance (km)': total_distance,
        'Average Utilization (%)': avg_util,
        'Avg Deliveries/Trip': avg_delivs,
        'Avg Weight/Trip (kg)': avg_weight_trip,
        'Density (kg/km)': density,
        'Total CO2 (kg)': total_co2,
        'CO2 per kg delivered (kg/kg)': co2_per_kg
    }
    perf_df = pd.DataFrame(perf.items(), columns=['Metric','Value'])
    print(perf_df.to_string(index=False))

    # Save outputs
    with pd.ExcelWriter('consolidation_outputs.xlsx') as writer:
        df_opt.to_excel(writer, sheet_name='consolidation', index=False)
        util.to_excel(writer, sheet_name='utilization', index=False)
        perf_df.to_excel(writer, sheet_name='performance', index=False)
        dist_map[['trip','vehicle_name','roundtrip_km','co2_kg']].to_excel(writer, sheet_name='distance_co2', index=False)

    # Visualize Performance
    plt.figure(figsize=(6,3))
    plt.axis('off')
    tbl = plt.table(cellText=perf_df.values, colLabels=perf_df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1,2)
    plt.title('Performance & Sustainability Summary')
    plt.tight_layout()
    plt.savefig('performance_sustainability_summary.png')
    plt.show()

    # Utilization Chart
    plt.figure(figsize=(8,6))
    for t in sorted(util['trip'].unique()):
        sub = util[util['trip']==t]
        plt.bar(sub['vehicle_name']+f' (Trip {t})', sub['util_pct'])
    plt.xlabel('Vehicle (Trip)')
    plt.ylabel('Utilization (%)')
    plt.title('Vehicle Utilization per Trip')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('utilization_per_trip.png')
    plt.show()

    # Deliveries per Trip Histogram
    plt.figure(figsize=(8,6))
    df_opt.groupby('trip')['outlet'].nunique().plot(kind='bar')
    plt.xlabel('Trip')
    plt.ylabel('Distinct Outlets Served')
    plt.title('Deliveries per Trip')
    plt.tight_layout()
    plt.savefig('deliveries_per_trip.png')
    plt.show()

if __name__=='__main__':
    main()
