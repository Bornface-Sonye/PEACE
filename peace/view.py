from django.shortcuts import render, redirect
from django.views import View
from .forms import AnswerForm, InterrogatorReportForm
from .models import SuspectResponse

class FormsView(View):
    template_name = 'forms.html'
    
    def get(self, request):
        answer_form = AnswerForm()
        report_form = InterrogatorReportForm()
        return render(request, self.template_name, {'answer_form': answer_form, 'report_form': report_form})

    def post(self, request):
        answer_form = AnswerForm(request.POST)
        report_form = InterrogatorReportForm(request.POST)

        if answer_form.is_valid():
            case_description = answer_form.cleaned_data['case_description']
            unique_id = answer_form.cleaned_data['unique_id']
            trace = answer_form.cleaned_data['trace']
            know_complainant = answer_form.cleaned_data['know_complainant']
            involved_with_complainant = answer_form.cleaned_data['involved_with_complainant']
            recidivist = answer_form.cleaned_data['recidivist']
            
            serial_number = generate_serial_number(unique_id, case_description)
            suspectResponse = answer_form.save(commit=False)
            suspectResponse.serial_number = serial_number
            suspectResponse.save()

            return redirect('success', serial_number=serial_number)
        
        if report_form.is_valid():
            serial_number = report_form.cleaned_data['serial_number']
            try:
                suspect_response = get_object_or_404(SuspectResponse, serial_number=serial_number)
                suspect = suspect_response.unique_id
                # Your processing for generating report data
                # This part is incomplete as it involves external services or algorithms
                
            except SuspectResponse.DoesNotExist:
                return render(request, 'interrogator_error.html', {'error_message': f'Suspect Report with serial number "{serial_number}" not found.'})
            
            
            mlm = MachineLearningModel()
            accuracy = mlm.accuracy()
            sent1 = SentimentAnalyser()
            criminal = CriminalPrediction()
            name = suspect.unique_id
            age = suspect.age
            gender = suspect.gender
            recidivist = suspect_response.recidivist
            firstResponse = suspect_response.trace
            secondResponse = suspect_response.know_complainant
            consistency_score = sent1.calculate_consistency_score(firstResponse, secondResponse)
            trace = sent1.is_obedient(firstResponse, name, age, gender)
            obedient_score = sent1.is_obedient(firstResponse, name, age, gender)
            criminal.data_retrieval(name, age, recidivist, trace, obedient_score, consistency_score, gender)
            criminal.data_preparation()
            result = criminal.result()
            
            report_data = {
                    'accuracy': accuracy,
                    'name': name,
                    'case_description': suspect_response.case_description,
                    'result': result,
                    
            }
            return render(request, 'report.html', {'report_data': report_data})
        
        
        
        return render(request, self.template_name, {'answer_form': answer_form, 'report_form': report_form})
