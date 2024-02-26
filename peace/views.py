<<<<<<< HEAD
from django.views import View
from .forms import AnswerForm,InterrogatorReportForm
from django.shortcuts import render, redirect,get_object_or_404
from .models import Suspect, Case, SuspectTestification
from django.http import HttpResponse
from django.conf import settings
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer
from reportlab.lib.units import inch
from io import BytesIO

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy as sp
from django.conf import settings


class SentimentAnalyser:
    def __init__(self):
        # Initialize NLTK Sentiment Intensity Analyzer
        self.sid = SentimentIntensityAnalyzer()
        # Initialize spaCy NLP model
        self.nlp = sp.load("en_core_web_sm")
    
    def is_obedient(self, answer_1, name, age, gender):
        # Sentiment analysis for obedience
        # You can define your own rules for determining obedience based on sentiment scores
        sentiment_scores = self.sid.polarity_scores(answer_1)
        obedient_score = sentiment_scores['compound']
        if obedient_score >= 0:
            return 'Yes' # Obedient
        else:
            return 'No' # Not obedient
        
    def calculate_emotion_score(self, answer_1):
        # Calculate emotion score
        sentiment_scores = self.sid.polarity_scores(answer_1)
        emotion_score = sentiment_scores['compound']
        return emotion_score

    def calculate_consistency_score(self, answer_1, answer_2):
        # Calculate consistency score based on two sets of text
        doc1 = self.nlp(answer_1)
        doc2 = self.nlp(answer_2)
        # You can define your own logic to calculate consistency score
        # For example, you can measure the similarity between two text samples
        similarity = doc1.similarity(doc2)
        consistency_score = (similarity + 1) / 2 # Normalize to [0, 1]
        return consistency_score

    def calculate_confidence_score(self, emotion_score, consistency_score):
        # Calculate confidence score as an average of emotion and consistency scores
        confidence_score = (emotion_score + consistency_score) / 2
        return confidence_score

class MachineLearningModel:
    def __init__(self):
        self.new_data = None
        self.prediction = None
        self.numeric_feature = None
        self.categorical_feature = None
        self.training_features = None
        self.predictions = None
        self.model = None
        self.scaler = None
        self.actual_labels = None
        self.file_path = os.path.join(settings.BASE_DIR, 'peace', 'peace', 'crime.csv')
        self.df = pd.read_csv(self.file_path)
        self.feature_names = ['Age','DrugTest','Obedient','EmotionScore','ConfidenceScore','ConsistencyScore','Gender']
        self.training_features = self.df[self.feature_names]
        self.outcome_name = ['Criminal']
        self.outcome_labels = self.df[self.outcome_name]
        self.numeric_feature_names = ['Age','EmotionScore','ConfidenceScore','ConsistencyScore']
        self.categorical_feature_names = ['DrugTest','Obedient','Gender']
        ss = StandardScaler()
        ss.fit(self.training_features[self.numeric_feature_names])
        self.training_features[self.numeric_feature_names] = ss.transform(self.training_features[self.numeric_feature_names])
        self.training_features = pd.get_dummies(self.training_features,columns=self.categorical_feature_names)
        self.categorical_engineered_features = list(set(self.training_features.columns)-set(self.numeric_feature_names))
        self.lr = LogisticRegression()
        self.model = self.lr.fit(self.training_features,np.array(self.outcome_labels['Criminal']))
        self.pred_labels = self.model.predict(self.training_features)
        self.actual_labels = np.array(self.outcome_labels['Criminal'])
        if not os.path.exists('Model'):
            os.mkdir('Model')
        if not os.path.exists('Scaler'):
            os.mkdir('Scaler')
        joblib.dump(self.model,r'Model/model.pickle')
        joblib.dump(ss,r'Scaler/scaler.pickle')
        self.model = joblib.load(r'Model/model.pickle')
        self.scaler = joblib.load(r'Scaler/scaler.pickle')

    def accuracy(self):
        acc = accuracy_score(self.actual_labels, self.pred_labels)
        class_stats = classification_report(self.actual_labels, self.pred_labels)

        accuracy_row = f"Accuracy: {acc * 100:.2f}%"
        class_stats_row = "Classification Stats:\n" + class_stats

        return accuracy_row

class CriminalPrediction:
    def __init__(self):
        self.new_data = None
        self.prediction = None
        self.training_features = None
        self.numeric_feature = None
        self.categorical_feature = None
        self.model = joblib.load(r'Model/model.pickle')
        self.scaler = joblib.load(r'Scaler/scaler.pickle')
        self.prediction = None
        self.predictions = None
        self.file_path = os.path.join(settings.BASE_DIR, 'peace', 'peace', 'crime.csv')
        self.df = pd.read_csv(self.file_path)
        self.feature_names = ['Age','DrugTest','Obedient','EmotionScore','ConfidenceScore','ConsistencyScore','Gender']
        self.training_features = self.df[self.feature_names]
        self.outcome_name = ['Criminal']
        self.outcome_labels =self.df[self.outcome_name]
        self.numeric_feature_names = ['Age','EmotionScore','ConfidenceScore','ConsistencyScore']
        self.categorical_feature_names = ['DrugTest','Obedient','Gender']
        self.ss = StandardScaler()
        self.ss.fit(self.training_features[self.numeric_feature_names])
        self.training_features[self.numeric_feature_names] = self.ss.transform(self.training_features[self.numeric_feature_names])
        self.training_features = pd.get_dummies(self.training_features,columns=self.categorical_feature_names)
        
    
    def data_retrieval(self,name,age,drug_test,obedient,emotion_score,confidence_score,consistency_score,gender):
        self.new_data = pd.DataFrame([{'Name': name,
                        'Age': age,
                        'DrugTest': drug_test,
                        'Obedient': obedient,
                        'EmotionScore': emotion_score,
                        'ConfidenceScore': confidence_score,
                        'ConsistencyScore': consistency_score,
                        'Gender': gender
                        }])
    
    def data_preparation(self):
        self.prediction = self.new_data.copy()
        self.numeric_feature = ['Age','EmotionScore','ConfidenceScore','ConsistencyScore']
        self.categorical_feature = ['DrugTest_Yes','DrugTest_No','Obedient_Yes','Obedient_No','Gender_Male','Gender_Female']
        self.prediction[self.numeric_feature] = self.scaler.transform(self.prediction[self.numeric_feature])
        self.prediction= pd.get_dummies(self.prediction,columns=['DrugTest','Obedient','Gender'])
        for feature in self.categorical_feature:
            if feature not in self.prediction.columns:
                self.prediction[feature] = 0 #Add missing categorical feature columns with 0 columns
        self.prediction = self.prediction[self.training_features.columns]
        self.predictions = self.model.predict(self.prediction)
        self.new_data['Criminal'] = self.predictions
    
    def result(self):
        result_str = "Prediction Result:\n"
        for index, row in self.new_data.iterrows():
            result_str += f"Name: {row['Name']}, Age: {row['Age']}, Gender: {row['Gender']}, Predicted Criminal: {'Yes' if row['Criminal'] == 1 else 'No'}\n"
        return result_str


class ErrorPageView(View):
    def get(self,request):
        return render(request, 'interrogator_error.html')

class SuccessPageView(View):
    def get(self,request):
        return render(request, 'interrogator_success.html')

class HomePageView(View):
    def get(self,request):
        return render(request, 'index.html')

class InterrogatorDashboardView(View):
    def get(self,request):
        return render(request, 'interrogator_dashboard.html')

class AddAnswerView(View):
    template_name = 'answer_form.html'
    
    def get(self, request):
        form = AnswerForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = AnswerForm(request.POST)

        #Ensure that every entry is valid
        if form.is_valid():
            email_address = form.cleaned_data['email_address']
            case_description = form.cleaned_data['case_description']
            answer_1 = form.cleaned_data['answer_1']
            answer_2 = form.cleaned_data['answer_2']

            # Save the Interrogation Answers
            statement = form.save(commit=False)
            statement.save()

            return redirect('success')
        else:
            return render(request, self.template_name, {'form': form})

class InterrogatorReportView(View):
    template_name = 'interrogator_report.html'

    def get(self, request):
        form = InterrogatorReportForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = InterrogatorReportForm(request.POST)

        email_address = request.POST.get('email_address')
        try:
            testification = SuspectTestification.objects.get(email_address=email_address)
            suspect = Suspect.objects.get(email_address=email_address)            
        except SuspectTestification.DoesNotExist:
            return render(request, 'interrogator_error.html')
        
        #Supervised Learning
        mlm = MachineLearningModel()
        accuracy = mlm.accuracy()
        #Sentiment Analysis
        sent1 = SentimentAnalyser()
        # Prediction
        criminal = CriminalPrediction()
        name = testification.email_address
        age = suspect.age
        drug_test = suspect.drug_test
        answer_1 = testification.answer_1
        answer_2 = testification.answer_2
        gender = suspect.gender
        obedient_score = sent1.is_obedient(answer_1,name,age,gender)
        emotion_score = sent1.calculate_emotion_score(answer_1)
        consistency_score = sent1.calculate_consistency_score(answer_1,answer_2)
        confidence_score = sent1.calculate_confidence_score(emotion_score,consistency_score)
        criminal.data_retrieval(name,age,drug_test,obedient_score,emotion_score,confidence_score,consistency_score,gender)
        criminal.data_preparation()
        result = criminal.result()

        report_data = {
            'accuracy':accuracy,
            'suspect_details': {
                'email_address': name,
            },
            'case_description': testification.case_description,
            'result':result,
        }

        # Generate PDF
        pdf_bytes = generate_pdf(report_data)

        # Return the PDF file as a response
        response = HttpResponse(pdf_bytes, content_type='interrogator/pdf')
        response['Content-Disposition'] = f'attachment; filename="{email_address}_report.pdf"'
        return response 

def generate_pdf(report_data):
    buffer = BytesIO()
    pdf_canvas = SimpleDocTemplate(buffer, pagesize=landscape(letter))

    # Set font and size
    styles = getSampleStyleSheet()
    style_normal = styles['Normal']
    style_heading = styles['Heading1']
    style_normal.fontName = 'Helvetica'
    style_heading.fontName = 'Helvetica'
    style_normal.fontSize = 12
    style_heading.fontSize = 14

    # Define table data
    table_data = [
        ["Suspect Report"],
        ["Email Address:", report_data['suspect_details']['email_address']],
        ["Case Description:", report_data['case_description']],
        ["Accuracy:", report_data['accuracy']],
        ["Result:", report_data['result']],
    ]

    # Create a table
    table = Table(table_data)

    # Set table style
    style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),  # Header background color
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),  # Content background color
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 12),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('SPAN', (0, 0), (1, 0)),  # Merge cells for the heading
                        ])

    table.setStyle(style)

    # Add page border
    frame_styling = TableStyle([('BOX', (0, 0), (-1, -1), 2, colors.black)])
    table.setStyle(frame_styling)

    # Add table to the PDF
    elements = [table]

    # Add space after table
    elements.append(Spacer(1, 0.25 * inch))

    # Build PDF
    pdf_canvas.build(elements)

    # Save the PDF file
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
=======
import datetime
from django.http import HttpResponse
from django.views import View
from django.shortcuts import render, redirect

def home(request):
    return render(request, 'index.html')

# views.py
from .forms import StatementForm

# views.py
from django.shortcuts import render, redirect
from .forms import WitnessTestificationForm
from .models import Witness
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import StatementForm
from .models import Statement, Enforcer, Case

def statement_form(request):
    if request.method == 'POST':
        form = StatementForm(request.POST)
        if form.is_valid():
            # Retrieve form data
            enforcer_id = form.cleaned_data['enforcer_id'].enforcer_id  # Extracting the enforcer_id
            case_description = form.cleaned_data['case_description']
            crime_location = form.cleaned_data['crime_location']
            crime_victim = form.cleaned_data['crime_victim']
            crime_incident_date = form.cleaned_data['crime_incident_date']
            crime_incident_time = form.cleaned_data['crime_incident_time']
            
            # Check if enforcer exists
            if not Enforcer.objects.filter(enforcer_id=enforcer_id).exists():
                messages.error(request, "Enforcer does not exist.")
                return render(request, 'statement_form.html', {'form': form})
            
            # Check if case description exists
            if not Case.objects.filter(case_description=case_description).exists():
                messages.error(request, "Case description does not exist.")
                return render(request, 'statement_form.html', {'form': form})
            
            # Check if statement already exists
            if Statement.objects.filter(enforcer_id=enforcer_id, case_description=case_description).exists():
                messages.error(request, "Statement already exists.")
                return render(request, 'statement_form.html', {'form': form})
            
            # Save the statement
            form.save()
            messages.success(request, "Statement added successfully.")
            return redirect('success_page')  # Redirect to a success page
    else:
        form = StatementForm()
    return render(request, 'statement_form.html', {'form': form})

def success_page(request):
    return render(request, 'index.html')  # Define your success page template

from django.shortcuts import render, redirect
from .forms import WitnessTestificationForm
from .models import Witness

def submit_testification(request):
    if request.method == 'POST':
        form = WitnessTestificationForm(request.POST)
        if form.is_valid():
            # Extract witness_id from the form data
            witness_id = form.cleaned_data['witness_id'].witness_id
            case_description = form.cleaned_data['case_description']
            # Check if witness_id and case_description exist
            if Witness.objects.filter(witness_id=witness_id).exists() and Case.objects.filter(case_description=case_description).exists():
                form.save()
                return render(request, 'index.html')
            else:
                return render(request, 'login.html', {'error_message': 'Invalid witness_id or case_description'})
    else:
        form = WitnessTestificationForm()
    return render(request, 'witness_testification_form.html', {'form': form})




# views.py
from django.shortcuts import render, redirect
from .forms import SuspectTestificationForm
from .models import Suspect, Case

def add_suspect_testification(request):
    if request.method == 'POST':
        form = SuspectTestificationForm(request.POST)
        if form.is_valid():
            # Extract suspect_id from the form data
            suspect_id = form.cleaned_data['suspect_id'].suspect_id
            case_description = form.cleaned_data['case_description']
            # Check if suspect_id and case_description exist
            if Suspect.objects.filter(suspect_id=suspect_id).exists() and Case.objects.filter(case_description=case_description).exists():
                form.save()
                return render(request, 'index.html')
            else:
                return render(request, 'login.html', {'error_message': 'Invalid suspect_id or case_description'})
    else:
        form = SuspectTestificationForm()
    return render(request, 'suspect_testification_form.html', {'form': form})

from django.shortcuts import render
from .models import Case, Suspect, Witness, Enforcer

def load_report(request):
    if request.method == 'POST':
        case_description = request.POST.get('case_description', None)
        if case_description:
            try:
                case = Case.objects.get(case_description=case_description)
                suspects = Suspect.objects.filter(case_description=case)
                witnesses = Witness.objects.filter(case_description=case)
                enforcers = Enforcer.objects.filter(case_description=case)
                return render(request, 'report.html', {'case': case, 'suspects': suspects, 'witnesses': witnesses, 'enforcers': enforcers})
            except Case.DoesNotExist:
                error_message = "Case not found for the provided description."
                return render(request, 'report.html', {'error_message': error_message})
        else:
            error_message = "Please provide a case description."
            return render(request, 'report.html', {'error_message': error_message})
    return render(request, 'report.html')




from django.shortcuts import render
from .models import Statement, WitnessTestification, Enforcer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def data_fusion(request):
    if request.method == 'POST':
        case_description = request.POST.get('case_description')
        
        # Query database for relevant data
        statements = Statement.objects.filter(case_description=case_description)
        witnesses = WitnessTestification.objects.filter(case_description=case_description)
        enforcers = Enforcer.objects.filter(case_description=case_description)
        
        # Load textual data from witnesses and enforcers
        witness_texts = [f"{witness.crime_location}, {witness.crime_victim}, {witness.crime_incident_date}, {witness.crime_incident_time}" for witness in witnesses]
        enforcer_texts = [f"{statement.crime_location}, {statement.crime_victim}, {statement.crime_incident_date}, {statement.crime_incident_time}" for statement in statements]
        
        # Combine witness and enforcer texts
        all_texts = witness_texts + enforcer_texts
        
        # Convert text data to numerical representation using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between each pair of documents
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Perform data fusion
        fused_data = {}
        for i, witness_text in enumerate(witness_texts):
            fused_data[witness_text] = []
            for j, enforcer_text in enumerate(enforcer_texts):
                fused_data[witness_text].append({
                    'enforcer_statement': enforcer_text,
                    'similarity_score': similarity_matrix[i, len(witness_texts) + j]
                })
        
        # Render HTML template with fused data
        return render(request, 'fused_data.html', {'fused_data': fused_data})
    else:
        return render(request, 'template_name.html')













>>>>>>> 7ff2cbcfdbc7181a4fe291243adbdb7c4db75281
