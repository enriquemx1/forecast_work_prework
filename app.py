import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io

# ------------------------------
# Title & Introduction
# ------------------------------
st.title("Time Series Forecasting with Prophet")
st.write("""
This application allows you to:
- Upload a CSV file with Monthly historical Dates and Calls.
- Select which columns correspond to the date and the target variable (y).
- Confirm that the file loaded correctly and in the correct format.
- Process the data and run a forecasting model using Prophet (optimized for monthly data).
- View the forecast (filtered from 5 months before the last observation until 3 months in the future) as a table and with graphs.
- Download the forecast in Excel format.
""")

# ------------------------------
# 1. Upload the CSV File
# ------------------------------
uploaded_file = st.file_uploader("Select the CSV file with your historical data", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File loaded successfully. Here is a preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading the file: {e}")
    
    # ------------------------------
    # 2. Column Selection: Date and Target Variable
    # ------------------------------
    col_options = df.columns.tolist()
    st.write("**Select the columns that correspond to the Date and the target variable (y):**")
    date_col = st.selectbox("Date Column:", col_options)
    y_col = st.selectbox("Target Variable Column (y):", col_options)
    
    if st.button("Confirm selection and process file"):
        try:
            # Convert the selected date column to datetime (using dayfirst=True for dd/mm/yyyy format)
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
            
            # Keep only the selected columns and rename them for Prophet
            df_prophet = df[[date_col, y_col]].rename(columns={date_col: "ds", y_col: "y"}).sort_values("ds")
            
            # Validate that the target variable 'y' is numeric
            if not pd.api.types.is_numeric_dtype(df_prophet["y"]):
                st.error("The selected column for 'y' is not numeric. Please check your data.")
            else:
                st.success("File processed and columns selected correctly.")
                st.write("Here is a preview of the data to be used:")
                st.dataframe(df_prophet.head())

                # ------------------------------
                # 3. Train the Prophet Model
                # ------------------------------
                model = Prophet(
                    growth='linear',
                    changepoint_prior_scale=0.05,
                    seasonality_mode='additive',
                    seasonality_prior_scale=5,
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    n_changepoints=10
                )
                with st.spinner("Training the model..."):
                    model.fit(df_prophet)
                
                # ------------------------------
                # 4. Generate Forecast
                # ------------------------------
                # Forecast 3 months into the future using a monthly start frequency ('MS')
                future = model.make_future_dataframe(periods=3, freq='MS')
                forecast = model.predict(future)
                
                # ------------------------------
                # 5. Define the Date Range for Visualization and Export
                # ------------------------------
                # Visualization range: 5 months before the last observation until 3 months in the future.
                last_date = df_prophet['ds'].max()
                start_date = last_date - pd.DateOffset(months=5)
                end_date = last_date + pd.DateOffset(months=3)
                
                # Filter the forecast to the desired date range and create a formatted date column
                forecast_filtered = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)].copy()
                forecast_filtered['Date'] = forecast_filtered['ds'].dt.strftime('%d/%m/%Y')
                
                # Select columns to export: formatted date, forecast, and forecast intervals
                df_export = forecast_filtered[['Date', 'yhat', 'yhat_lower', 'yhat_upper']]
                
                st.write("## Generated Forecast")
                st.dataframe(df_export)
                
                # ------------------------------
                # 6. Export the Forecast to Excel
                # ------------------------------
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df_export.to_excel(writer, index=False, sheet_name="Forecast")
                    writer.save()
                processed_data = output.getvalue()
                
                st.download_button(
                    label="Download Forecast as Excel",
                    data=processed_data,
                    file_name="forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # ------------------------------
                # 7. Display Forecast Graphs
                # ------------------------------
                st.write("### Forecast Plot")
                fig1 = model.plot(forecast)
                ax = fig1.gca()  # Get the Axes object to customize the plot
                ax.set_title('Prophet Forecast (Monthly Data)')
                ax.set_xlabel('Date')
                ax.set_ylabel(y_col)
                ax.set_xlim([start_date, end_date])
                st.pyplot(fig1)
                
                st.write("### Model Components")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)
                
                # ------------------------------
                # 8. Explanation of the Graphs
                # ------------------------------
                st.markdown("""
                **How to Interpret the Graphs:**
                
                - **Forecast Plot:**  
                  The central line (yhat) represents the predicted value of the target variable over time.
                  The shaded bands (yhat_lower and yhat_upper) display the uncertainty intervals.
                  The black dots represent the historical data points used to train the model.
                
                - **Model Components:**  
                  The panels show the individual components that contribute to the forecast such as the overall trend and seasonal effects (e.g., yearly seasonality).
                
                Use these graphs to understand the past behavior of the time series and the model's projection into the future.
                """)
        except Exception as ex:
            st.error(f"An error occurred while processing the file: {ex}")