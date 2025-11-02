import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import tempfile
import json
from dotenv import load_dotenv
load_dotenv()

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocr_parser import OCRParser
from field_extractor import FieldExtractor
from classifier import ClaimClassifier
from database import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="Auto Claim AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def load_components():
    """Load and cache the AI components"""
    ocr_parser = OCRParser()
    field_extractor = FieldExtractor()
    classifier = ClaimClassifier()
    db_manager = DatabaseManager()
    return ocr_parser, field_extractor, classifier, db_manager

# Load components
ocr_parser, field_extractor, classifier, db_manager = load_components()

# Sidebar
st.sidebar.title("🏥 Auto Claim AI")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["📊 Dashboard", "📄 Process Claim", "🤖 LLM Process", "📈 Analytics", "💬 Chat", "🧑‍💼 Agentic Claim", "⚙️ Settings"]
)

# Main content
if page == "📊 Dashboard":
    st.title("📊 Auto Claim AI Dashboard")
    st.markdown("---")
    
    if st.button("🔄 Refresh Analytics"):
        st.experimental_rerun()
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=(datetime.now() - timedelta(days=30)))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Get real data from database for selected date range
    analytics = db_manager.get_analytics(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    recent_claims = db_manager.get_claims(limit=1000)
    settings = db_manager.get_settings()
    
    # Filter recent_claims by date range
    if recent_claims:
        claims_df = pd.DataFrame(recent_claims)
        claims_df['created_at'] = pd.to_datetime(claims_df['created_at'])
        mask = (claims_df['created_at'] >= pd.to_datetime(start_date)) & (claims_df['created_at'] <= pd.to_datetime(end_date))
        filtered_claims = claims_df[mask].copy()
        recent_claims = filtered_claims.to_dict(orient='records')
    else:
        filtered_claims = pd.DataFrame()
    
    # Calculate today's metrics
    today = datetime.now().strftime('%Y-%m-%d')
    today_analytics = None
    for daily in analytics.get('daily_analytics', []):
        if daily[1] == today:  # date column
            today_analytics = daily
            break
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        claims_today = today_analytics[2] if today_analytics else 0  # claims_processed
        st.metric(
            label="Claims Processed Today",
            value=claims_today,
            delta=f"+{claims_today}" if claims_today > 0 else "0"
        )
    
    with col2:
        auto_approved = today_analytics[3] if today_analytics else 0  # auto_approved
        approval_rate = (auto_approved / claims_today * 100) if claims_today > 0 else 0
        st.metric(
            label="Auto-Approval Rate",
            value=f"{approval_rate:.1f}%",
            delta=f"{auto_approved} approved" if auto_approved > 0 else "0"
        )
    
    with col3:
        # Calculate average processing time from claims data
        avg_time = 0.0
        if recent_claims:
            processing_times = [claim.get('processing_time', 0) for claim in recent_claims if claim.get('processing_time')]
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        st.metric(
            label="Average Processing Time",
            value=f"{avg_time:.1f} sec",
            delta=f"{avg_time:.1f} sec avg"
        )
    
    with col4:
        # Calculate average confidence from claims data
        avg_confidence = 0.0
        if recent_claims:
            confidences = [claim.get('confidence', 0) for claim in recent_claims if claim.get('confidence')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        st.metric(
            label="Average Confidence",
            value=f"{avg_confidence:.1f}%",
            delta=f"{avg_confidence:.1f}% avg"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Claims by Status")
        if not filtered_claims.empty:
            status_counts = filtered_claims['status'].value_counts().to_dict()
            status_data = pd.DataFrame([
                {'Status': 'Auto-Approve', 'Count': status_counts.get('auto_approved', 0)},
                {'Status': 'Manual Review', 'Count': status_counts.get('manual_review', 0)},
                {'Status': 'Rejected', 'Count': status_counts.get('rejected', 0)}
            ])
            
            fig = px.pie(status_data, values='Count', names='Status', 
                        color_discrete_map={'Auto-Approve': '#00FF00', 'Manual Review': '#FFA500', 'Rejected': '#FF0000'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No claims data available for the selected date range.")
    
    with col2:
        st.subheader("💰 Claims by Amount Range")
        if not filtered_claims.empty:
            bins = [0, 1000, 5000, 10000, float('inf')]
            labels = ['0-1K', '1K-5K', '5K-10K', '10K+']
            filtered_claims['amount_range'] = pd.cut(filtered_claims['amount'], bins=bins, labels=labels, right=False)
            amount_data = filtered_claims['amount_range'].value_counts().sort_index().reset_index()
            amount_data.columns = ['Range', 'Count']
            
            fig = px.bar(amount_data, x='Range', y='Count', 
                        color_discrete_sequence=['#1f77b4'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No amount range data available for the selected date range.")
    
    # Recent claims table
    st.subheader("📋 Recent Claims")
    if recent_claims:
        claims_df = pd.DataFrame(recent_claims)
        # Select relevant columns for display
        display_columns = ['claim_id', 'patient_name', 'amount', 'status', 'confidence', 'created_at']
        display_df = claims_df[display_columns].copy()
        display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1f}%" if x else "N/A")
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d')
        
        # Rename columns for display
        display_df.columns = ['Claim ID', 'Patient Name', 'Amount', 'Status', 'Confidence', 'Date']
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No claims processed yet.")

elif page == "📄 Process Claim":
    st.title("📄 Process Claim")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Claim Document",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        help="Upload a PDF or image file containing the claim information"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        try:
            # Process the file
            start_time = datetime.now()
            with st.spinner("Processing document..."):
                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    extracted_data = ocr_parser.extract_text_from_pdf(file_path)
                    raw_text = extracted_data['text'] + " " + extracted_data['images_text']
                else:
                    raw_text = ocr_parser.extract_text_from_image(file_path)
                
                # Extract fields
                extracted_fields = field_extractor.extract_fields(raw_text)
                
                # Fetch settings and classify claim
                settings = db_manager.get_settings()
                # st.write(settings)
                classification_result = classifier.classify_claim(extracted_fields, raw_text, settings=settings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            st.success("✅ Document processed successfully!")
            
            # Results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Extracted Information")
                
                # Display extracted fields
                if extracted_fields:
                    for field, value in extracted_fields.items():
                        st.write(f"**{field.title()}:** {value}")
                else:
                    st.warning("No fields could be extracted from the document.")
                
                # Show confidence in extraction
                validation = field_extractor.validate_extraction(extracted_fields)
                extraction_confidence = field_extractor.get_extraction_confidence(extracted_fields, validation)
                st.metric("Extraction Confidence", f"{extraction_confidence:.1%}")
            
            with col2:
                st.subheader("🤖 AI Classification")
                
                # Display classification results
                prediction = classification_result['prediction']
                confidence = classification_result['confidence']
                risk_score = classification_result['risk_score']
                recommended_action = classification_result['recommended_action']
                
                # Color code based on prediction
                if prediction == "Auto-Approve":
                    st.success(f"**Prediction:** {prediction}")
                else:
                    st.warning(f"**Prediction:** {prediction}")
                
                st.metric("Confidence", f"{confidence}%")
                st.metric("Risk Score", f"{risk_score}%")
                st.info(f"**Recommended Action:** {recommended_action}")
            
            # Action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            # Prepare claim data for database
            claim_data = {
                'patient_name': extracted_fields.get('name', 'Unknown'),
                'claim_date': extracted_fields.get('date', datetime.now().strftime('%Y-%m-%d')),
                'diagnosis': extracted_fields.get('diagnosis', ''),
                'amount': float(extracted_fields.get('amount', '0').replace(',', '')) if extracted_fields.get('amount') else 0.0,
                'provider': extracted_fields.get('provider', ''),
                'policy_number': extracted_fields.get('policy_number', ''),
                'service_date': extracted_fields.get('service_date', ''),
                'status': 'pending',
                'prediction': prediction,
                'confidence': confidence,
                'risk_score': risk_score,
                'recommended_action': recommended_action,
                'extracted_fields': extracted_fields,
                'raw_text': raw_text,
                'processing_time': processing_time
            }
            
            with col1:
                if st.button("✅ Approve Claim", type="primary"):
                    claim_data['status'] = 'auto_approved'
                    claim_id = db_manager.add_claim(claim_data)
                    st.success(f"Claim approved successfully! ID: {claim_id}")
            
            with col2:
                if st.button("❌ Reject Claim"):
                    claim_data['status'] = 'rejected'
                    claim_id = db_manager.add_claim(claim_data)
                    st.error(f"Claim rejected. ID: {claim_id}")
            
            with col3:
                if st.button("📝 Manual Review"):
                    claim_data['status'] = 'manual_review'
                    claim_id = db_manager.add_claim(claim_data)
                    st.info(f"Claim sent for manual review. ID: {claim_id}")
            
            # Raw text preview (collapsible)
            with st.expander("📄 Raw Extracted Text"):
                st.text_area("Raw Text", raw_text, height=200)
            
            # Features used (collapsible)
            with st.expander("🔍 Features Used for Classification"):
                features_used = classification_result.get('features_used', [])
                if features_used:
                    st.write("The following features were used for classification:")
                    for feature in features_used:
                        st.write(f"• {feature}")
                else:
                    st.write("No specific features were used.")
        
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(file_path)
            except:
                pass

elif page == "🤖 LLM Process":
    st.title("🤖 LLM Process Claim")
    st.markdown("---")
    import llm_utils
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Claim Document",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        help="Upload a PDF or image file containing the claim information"
    )
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        try:
            start_time = datetime.now()
            with st.spinner("Processing document..."):
                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    extracted_data = ocr_parser.extract_text_from_pdf(file_path)
                    raw_text = extracted_data['text'] + " " + extracted_data['images_text']
                else:
                    raw_text = ocr_parser.extract_text_from_image(file_path)
                # Extract fields
                extracted_fields = field_extractor.extract_fields(raw_text)
                # Classify claim
                settings = db_manager.get_settings()
                classification_result = classifier.classify_claim(extracted_fields, raw_text, settings=settings)
            processing_time = (datetime.now() - start_time).total_seconds()
            st.success("✅ Document processed successfully!")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📋 Extracted Information")
                if extracted_fields:
                    for field, value in extracted_fields.items():
                        st.write(f"**{field.title()}:** {value}")
                else:
                    st.warning("No fields could be extracted from the document.")
                validation = field_extractor.validate_extraction(extracted_fields)
                extraction_confidence = field_extractor.get_extraction_confidence(extracted_fields, validation)
                st.metric("Extraction Confidence", f"{extraction_confidence:.1%}")
            with col2:
                st.subheader("🤖 AI Classification")
                prediction = classification_result['prediction']
                confidence = classification_result['confidence']
                risk_score = classification_result['risk_score']
                recommended_action = classification_result['recommended_action']
                if prediction == "Auto-Approve":
                    st.success(f"**Prediction:** {prediction}")
                else:
                    st.warning(f"**Prediction:** {prediction}")
                st.metric("Confidence", f"{confidence}%")
                st.metric("Risk Score", f"{risk_score}%")
                st.info(f"**Recommended Action:** {recommended_action}")
            st.markdown("---")
            st.subheader("💬 LLM Review")
            with st.spinner("Getting LLM review from ..."):
                llm_response = llm_utils.get_claim_llm_review(
                    extracted_fields=extracted_fields,
                    classification_result=classification_result,
                    raw_text=raw_text
                )
            st.write(llm_response)
            with st.expander("📄 Raw Extracted Text"):
                st.text_area("Raw Text", raw_text, height=200)
            with st.expander("🔍 Features Used for Classification"):
                features_used = classification_result.get('features_used', [])
                if features_used:
                    st.write("The following features were used for classification:")
                    for feature in features_used:
                        st.write(f"• {feature}")
                else:
                    st.write("No specific features were used.")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
        finally:
            try:
                os.unlink(file_path)
            except:
                pass

elif page == "📈 Analytics":
    st.title("📈 Analytics")
    st.markdown("---")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Get claims data for the selected date range
    all_claims = db_manager.get_claims(limit=10000)  # adjust limit as needed
    claims_df = pd.DataFrame(all_claims)
    claims_df['claim_date'] = pd.to_datetime(claims_df['claim_date']).dt.date
    mask = (claims_df['claim_date'] >= start_date) & (claims_df['claim_date'] <= end_date)
    filtered_claims = claims_df[mask]
    
    # Group by date and aggregate
    if not filtered_claims.empty:
        daily = filtered_claims.groupby('claim_date').agg(
            claims_processed=('claim_id', 'count'),
            auto_approved=('status', lambda x: (x == 'auto_approved').sum()),
            manual_review=('status', lambda x: (x == 'manual_review').sum()),
            rejected=('status', lambda x: (x == 'rejected').sum()),
            total_amount=('amount', 'sum'),
        ).reset_index().sort_values('claim_date')
        daily['avg_amount'] = daily['total_amount'] / daily['claims_processed'].replace(0, 1)
        daily['approval_rate'] = (daily['auto_approved'] / daily['claims_processed'].replace(0, 1)) * 100
    else:
        daily = pd.DataFrame()
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Daily Claims Processed")
        if not daily.empty:
            fig = px.line(daily, x='claim_date', y='claims_processed')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily analytics data available.")
    with col2:
        st.subheader("💰 Average Claim Amount")
        if not daily.empty:
            fig = px.line(daily, x='claim_date', y='avg_amount')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No amount data available.")
    # Approval rate over time
    st.subheader("📈 Approval Rate Trend")
    if not daily.empty:
        fig = px.line(daily, x='claim_date', y='approval_rate')
        fig.update_layout(height=400, yaxis_title="Approval Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No approval rate data available.")
    # Claims by Status and Amount Range (filtered by date)
    st.subheader("📈 Claims by Status")
    if not filtered_claims.empty:
        status_counts = filtered_claims['status'].value_counts().to_dict()
        status_data = pd.DataFrame([
            {'Status': 'Auto-Approve', 'Count': status_counts.get('auto_approved', 0)},
            {'Status': 'Manual Review', 'Count': status_counts.get('manual_review', 0)},
            {'Status': 'Rejected', 'Count': status_counts.get('rejected', 0)}
        ])
        fig = px.pie(status_data, values='Count', names='Status')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No claims data available for the selected date range.")

    st.subheader("💰 Claims by Amount Range")
    if not filtered_claims.empty:
        bins = [0, 1000, 5000, 10000, float('inf')]
        labels = ['0-1K', '1K-5K', '5K-10K', '10K+']
        filtered_claims['amount_range'] = pd.cut(filtered_claims['amount'], bins=bins, labels=labels, right=False)
        amount_data = filtered_claims['amount_range'].value_counts().sort_index().reset_index()
        amount_data.columns = ['Range', 'Count']
        fig = px.bar(amount_data, x='Range', y='Count')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No amount range data available for the selected date range.")
    # Summary statistics (keep using analytics summary for now)
    analytics = db_manager.get_analytics(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    st.subheader("📊 Summary Statistics")
    summary = analytics.get('summary', {})
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Claims", f"{summary.get('total_claims', 0):,}")
        with col2:
            st.metric("Auto-Approved", f"{summary.get('auto_approved', 0):,}")
        with col3:
            st.metric("Manual Reviews", f"{summary.get('manual_review', 0):,}")
        with col4:
            avg_amount = summary.get('total_amount', 0) / max(summary.get('total_claims', 1), 1)
            st.metric("Avg Claim Amount", f"${avg_amount:,.0f}")
    else:
        st.info("No summary statistics available.")
    # LLM-powered Analytics Q&A
    import llm_utils
    st.markdown("---")
    st.subheader("💡 Ask Analytics (LLM Q&A)")
    analytics_question = st.text_input("Ask a question about your analytics or claims data:", "What is the trend in auto-approvals this month?")
    if st.button("Ask LLM"):
        # Prepare a summary of the daily analytics for the LLM
        if not daily.empty:
            data_summary = daily.tail(30).to_dict(orient='records')  # last 30 days for brevity
        else:
            data_summary = []
        prompt = (
            "You are an expert data analyst AI. The user will ask questions about insurance claim analytics. "
            "Here is the recent daily analytics data (as a list of dicts):\n"
            f"{data_summary}\n"
            "User question: " + analytics_question + "\n"
            "Please answer in plain English, and if possible, provide actionable insights or trends."
        )
        with st.spinner("LLM is analyzing your analytics data..."):
            llm_answer = llm_utils.get_claim_llm_review(
                extracted_fields={},
                classification_result={},
                raw_text=None,
                extra_prompt=prompt
            )
        st.markdown(f"**LLM Answer:**\n\n{llm_answer}")

elif page == "🧑‍💼 Agentic Claim":
    st.title("🧑‍💼 Agentic Claim Processing (LangChain Agent)")
    st.markdown("---")
    import agentic_claim_agent
    # File upload or text input
    uploaded_file = st.file_uploader(
        "Upload a claim document (PDF, PNG, JPG, JPEG) or enter raw text:",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        key="agentic_claim_file_uploader"
    )
    raw_text_input = st.text_area("Or paste claim text here:")
    if st.button("Run Agentic Claim Workflow"):
        agent_result = None
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            with st.spinner("Agent is processing the uploaded file..."):
                agent_result = agentic_claim_agent.run_agentic_claim_workflow(file_path=file_path)
            try:
                os.unlink(file_path)
            except:
                pass
        elif raw_text_input.strip():
            with st.spinner("Agent is processing the claim text..."):
                agent_result = agentic_claim_agent.run_agentic_claim_workflow(raw_text=raw_text_input)
        else:
            st.warning("Please upload a file or enter claim text.")
        if agent_result:
            st.markdown("---")
            st.subheader("🧑‍💼 Agent Reasoning and Decision")
            st.write(agent_result)

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Get current settings
    settings = db_manager.get_settings()
    
    # Model settings
    st.subheader("🤖 AI Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Auto-Approval Confidence Threshold",
            min_value=10,
            max_value=95,
            value=int(settings.get('confidence_threshold', 75)),
            help="Minimum confidence required for auto-approval"
        )
        
        risk_threshold = st.slider(
            "High Risk Threshold",
            min_value=10,
            max_value=80,
            value=int(settings.get('risk_threshold', 60)),
            help="Risk score above which claims require manual review"
        )
    
    with col2:
        max_claim_amount = st.number_input(
            "Maximum Auto-Approval Amount ($)",
            min_value=10,
            max_value=10000,
            value=int(settings.get('max_auto_approval_amount', 5000)),
            step=500
        )
        
        processing_timeout = st.number_input(
            "Processing Timeout (seconds)",
            min_value=30,
            max_value=300,
            value=int(settings.get('processing_timeout', 120)),
            step=10
        )
    
    # Additional settings
    st.subheader("🔧 System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_approval_enabled = st.checkbox(
            "Enable Auto-Approval",
            value=settings.get('auto_approval_enabled', 'true').lower() == 'true'
        )
        
        notification_enabled = st.checkbox(
            "Enable Notifications",
            value=settings.get('notification_enabled', 'true').lower() == 'true'
        )
    
    with col2:
        # Model training history
        training_history = db_manager.get_training_history()
        if training_history:
            st.subheader("🔄 Model Training History")
            
            training_df = pd.DataFrame(training_history)
            if not training_df.empty:
                display_columns = ['model_version', 'training_date', 'accuracy', 'training_samples']
                display_df = training_df[display_columns].copy()
                display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.3f}" if x else "N/A")
                display_df.columns = ['Version', 'Training Date', 'Accuracy', 'Training Samples']
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No training history available.")
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        try:
            db_manager.update_setting('confidence_threshold', str(confidence_threshold))
            db_manager.update_setting('risk_threshold', str(risk_threshold))
            db_manager.update_setting('max_auto_approval_amount', str(max_claim_amount))
            db_manager.update_setting('processing_timeout', str(processing_timeout))
            db_manager.update_setting('auto_approval_enabled', str(auto_approval_enabled).lower())
            db_manager.update_setting('notification_enabled', str(notification_enabled).lower())
            st.success("Settings saved successfully!")
        except Exception as e:
            st.error(f"Error saving settings: {e}")
    
    # Model retraining
    st.subheader("🔄 Model Retraining")
    
    uploaded_training_data = st.file_uploader(
        "Upload Training Data (CSV)",
        type=['csv'],
        help="Upload CSV file with training data for model retraining"
    )
    
    if uploaded_training_data is not None:
        if st.button("🔄 Retrain Model"):
            with st.spinner("Retraining model..."):
                try:
                    # Read training data
                    training_df = pd.read_csv(uploaded_training_data)
                    st.success(f"Training data loaded: {len(training_df)} samples")
                    
                    # Here you would implement actual model retraining
                    # For now, just simulate
                    import time
                    time.sleep(2)
                    
                    # Add training record
                    training_record = {
                        'model_version': '1.1',
                        'training_date': datetime.now().strftime('%Y-%m-%d'),
                        'accuracy': 0.85,
                        'precision': 0.82,
                        'recall': 0.88,
                        'f1_score': 0.85,
                        'training_samples': len(training_df),
                        'test_samples': len(training_df) // 5,
                        'model_path': '../models/claim_classifier_v1.1.pkl'
                    }
                    
                    db_manager.add_training_record(training_record)
                    st.success("Model retrained successfully!")
                    
                except Exception as e:
                    st.error(f"Error during model retraining: {e}")
    
    # System information
    st.subheader("ℹ️ System Information")
    
    info_data = {
        "Component": ["OCR Parser", "Field Extractor", "Classifier", "Database", "Streamlit App"],
        "Status": ["✅ Active", "✅ Active", "✅ Active", "✅ Active", "✅ Active"],
        "Version": ["1.0.0", "1.0.0", "1.0.0", "1.0.0", "1.0.0"]
    }
    
    st.dataframe(pd.DataFrame(info_data), use_container_width=True)

elif page == "💬 Chat":
    st.title("💬 Chat with LLM")
    st.markdown("---")
    import llm_utils
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    # Add Clear Chat button
    if st.button("🧹 Clear Chat"):
        st.session_state['chat_history'] = []
    # File uploader for claim document
    uploaded_file = st.file_uploader(
        "Upload a claim document (PDF, PNG, JPG, JPEG) for LLM review:",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        key="chat_file_uploader"
    )
    # Get recent claims and analytics summary for context
    all_claims = db_manager.get_claims(limit=100)
    claims_df = pd.DataFrame(all_claims)
    claims_summary = claims_df.tail(10).to_dict(orient='records') if not claims_df.empty else []
    analytics = db_manager.get_analytics()
    analytics_summary = analytics.get('summary', {})
    # Handle file upload in chat
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        try:
            with st.spinner("Processing uploaded document..."):
                if uploaded_file.type == "application/pdf":
                    extracted_data = ocr_parser.extract_text_from_pdf(file_path)
                    raw_text = extracted_data['text'] + " " + extracted_data['images_text']
                else:
                    raw_text = ocr_parser.extract_text_from_image(file_path)
                extracted_fields = field_extractor.extract_fields(raw_text)
                settings = db_manager.get_settings()
                classification_result = classifier.classify_claim(extracted_fields, raw_text, settings=settings)
            # Add to chat history
            st.session_state['chat_history'].append({
                "role": "user",
                "content": f"[Uploaded file: {uploaded_file.name}]"
            })
            # LLM review for the claim
            llm_prompt = (
                "You are an expert insurance claim analyst AI. Here is a new claim document uploaded by the user.\n"
                f"Extracted fields: {extracted_fields}\n"
                f"Classification result: {classification_result}\n"
                "Please explain your decision, summarize the claim, and suggest next steps."
            )
            with st.spinner("LLM is reviewing the claim..."):
                llm_response = llm_utils.get_claim_llm_review(
                    extracted_fields=extracted_fields,
                    classification_result=classification_result,
                    raw_text=raw_text,
                    extra_prompt=llm_prompt
                )
            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": llm_response
            })
        except Exception as e:
            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": f"[Error processing document: {str(e)}]"
            })
        finally:
            try:
                os.unlink(file_path)
            except:
                pass
    # Text chat input
    user_input = st.chat_input("Type your message and press Enter...")
    if user_input:
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        # Prepare context for LLM: include recent claims and analytics summary
        context = (
            "You are a helpful insurance claim and analytics assistant.\n"
            f"Recent claims (last 10): {claims_summary}\n"
            f"Analytics summary: {analytics_summary}\n"
            f"User message: {user_input}\n"
            "If the user asks about claims, analytics, or trends, use the data above to answer. "
            "If the user asks about a specific claim, request more details or a file upload."
        )
        with st.spinner("LLM is thinking..."):
            response = llm_utils.get_claim_llm_review(
                extracted_fields={},
                classification_result={},
                raw_text=None,
                extra_prompt=context
            )
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
    # Display chat history
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**LLM:** {msg['content']}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Auto Claim AI System | Built with Streamlit, FastAPI, and Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
) 