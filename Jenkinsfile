pipeline {
    agent any

    environment {
        AWS_REGION = 'ap-south-1'  // Set your AWS region here
    }

    stages {
        stage('Trigger CodeBuild') {
            steps {
                script {
                    def codebuildProjectName = 'aidocker'
                    def awsRegion = env.AWS_REGION

                    sh """
                    aws codebuild start-build --project-name ${codebuildProjectName} --region ${awsRegion}
                    """
                }
            }
        }
    }
}
